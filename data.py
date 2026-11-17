"""
  python data.py record
  python data.py audit --kind e1m2
  python data.py unpack
  python data.py encode --tok tokenizer.pt
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from common import DATA, LEVELS, OUT

KINDS = [lv.lower() + h for h in ("", "hold") for lv in LEVELS]


def files(kind):
    return sorted(glob.glob(os.path.join(DATA, f"{kind}_*.npz")))


def _record(job):
    from game import record
    kind, seed, n, seg = job
    path = os.path.join(DATA, f"{kind}_{seed}.npz")
    if not os.path.exists(path):
        d = record(n, seed, kind[:4].upper(), seg=seg)
        mpath = os.path.join(DATA, "enemy", os.path.basename(path))
        np.savez_compressed(mpath + ".tmp.npz", masks=np.packbits(d.pop("masks"), -1))
        os.replace(mpath + ".tmp.npz", mpath)
        np.savez_compressed(path + ".tmp.npz", **d)
        os.replace(path + ".tmp.npz", path)
    return path


def record(args):
    from bot import Grid
    os.makedirs(os.path.join(DATA, "enemy"), exist_ok=True)
    for lv in LEVELS:
        Grid(lv)  # build the grids before the parallel jobs
    jobs = []
    for li, lv in enumerate(LEVELS):
        jobs += [(lv.lower(), 30000 + 1000 * li + k, args.frames, args.seg) for k in range(args.files)]
        jobs += [(lv.lower() + "hold", 39000 + 100 * li + k, 12000, 600) for k in range(args.hold)]
    t = time.perf_counter()
    with Pool(args.workers) as p:
        for i, path in enumerate(p.imap_unordered(_record, jobs)):
            print(f"{i + 1}/{len(jobs)} {os.path.basename(path)}  {time.perf_counter() - t:.0f} s", flush=True)


def audit(args):
    from game import inside, wad_map
    w = wad_map(args.kind[:4].upper())
    rng, rand, spawn, tot = np.random.default_rng(0), [], [], np.zeros(4)
    paths = files(args.kind)
    per_file = -(-48 // len(paths))
    for path in paths:
        d = np.load(path)
        f, pos, ep = d["frames"], d["pos"].astype(np.float64), d["episodes"]
        inmap = np.zeros(len(pos), bool)
        for ls in w["lines"]:
            if len(ls) >= 3:
                inmap |= inside(pos[:, 0], pos[:, 1], ls)
        y = f[:, 26:46, 16:112].astype(np.int16)  # game text: a thin yellow line in rows 34-36
        y = (y[..., 0] - y[..., 2] > 90) & (y[..., 1] - y[..., 2] > 60)
        text = (y[:, 8:11].sum((1, 2)) >= 12) & (y[:, :7].sum((1, 2)) + y[:, 12:].sum((1, 2)) <= 4)
        m = np.load(os.path.join(DATA, "enemy", os.path.basename(path)))["masks"]
        enemy = np.unpackbits(m, -1).reshape(len(f), -1).sum(1) >= 30
        tot += [len(f), (~inmap).sum(), text.sum(), enemy.sum()]
        print(f"{os.path.basename(path)}: outside the map {(~inmap).mean():6.2%}, game text {text.mean():6.2%}, "
              f"enemy on screen {enemy.mean():5.1%}", flush=True)
        rand.append(f[rng.choice(len(f), per_file, replace=False)])
        first = np.r_[0, np.nonzero(np.diff(ep))[0] + 1] + 5
        first = first[first < len(f)]
        spawn.append(f[rng.choice(first, min(per_file, len(first)), replace=False)])
    print(f"all {int(tot[0])} frames: outside the map {tot[1] / tot[0]:.2%}, game text {tot[2] / tot[0]:.2%}, "
          f"enemy on screen {tot[3] / tot[0]:.1%}")
    sheet = []
    for x in (np.concatenate(rand), np.concatenate(spawn)):
        x = x[rng.permutation(len(x))[:48]]
        x = np.concatenate([x, np.zeros((48 - len(x), *x.shape[1:]), np.uint8)])
        sheet.append(np.concatenate([np.concatenate(list(x[r * 12:(r + 1) * 12]), 1) for r in range(4)], 0))
    im = Image.fromarray(np.concatenate([sheet[0], np.full((8, sheet[0].shape[1], 3), 255, np.uint8), sheet[1]], 0))
    path = os.path.join(DATA, f"audit_{args.kind}.png")
    im.resize((im.width * 2, im.height * 2), Image.NEAREST).save(path)
    print(f"-> {path}: random frames on top, 5 frames after each spawn below")


def unpack(args):
    paths = [p for lv in LEVELS for p in files(lv.lower())]
    n = sum(len(np.load(p)["episodes"]) for p in paths)
    out = np.lib.format.open_memmap(os.path.join(DATA, "frames.tmp.npy"), "w+", np.uint8, (n, 96, 128, 3))
    masks = np.lib.format.open_memmap(os.path.join(DATA, "masks.tmp.npy"), "w+", np.uint8, (n, 96, 16))
    i = 0
    for p in paths:
        f = np.load(p)["frames"]
        out[i:i + len(f)] = f
        masks[i:i + len(f)] = np.load(os.path.join(DATA, "enemy", os.path.basename(p)))["masks"]
        i += len(f)
    out.flush()
    masks.flush()
    del out, masks
    for name in ("frames", "masks"):
        os.replace(os.path.join(DATA, f"{name}.tmp.npy"), os.path.join(DATA, f"{name}.npy"))
    print(f"wrote data/frames.npy and data/masks.npy: {n} frames from {len(paths)} files")


def encode(args):
    from model import encode_frames, load_tokenizer
    tok = load_tokenizer(args.tok)
    for kind in KINDS:
        t, lat, act, eps, frames, base = time.perf_counter(), [], [], [], [], 0
        for path in files(kind):
            d = np.load(path)
            lat.append(encode_frames(tok, d["frames"]))
            act.append(d["actions"])
            eps.append(d["episodes"] + base)
            base += int(d["episodes"].max()) + 1
            if kind.endswith("hold"):
                frames.append(d["frames"])
        out = os.path.join(OUT, f"lat_{kind}.npz")
        np.savez(out, latents=np.concatenate(lat), actions=np.concatenate(act), episodes=np.concatenate(eps),
                 **({"frames": np.concatenate(frames)} if frames else {}))
        print(f"{kind}: {sum(len(x) for x in lat)} frames in {time.perf_counter() - t:.0f} s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "audit", "unpack", "encode"])
    ap.add_argument("--files", type=int, default=25, help="record: training files per level")
    ap.add_argument("--frames", type=int, default=20000, help="record: frames per training file")
    ap.add_argument("--seg", type=int, default=1000, help="record: steps between spawns in the training files")
    ap.add_argument("--hold", type=int, default=2, help="record: held-out files per level")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--kind", default="e1m1", choices=KINDS, help="audit: which files")
    ap.add_argument("--tok", default="tokenizer.pt", help="encode: the tokenizer in OUT")
    args = ap.parse_args()
    {"record": record, "audit": audit, "unpack": unpack, "encode": encode}[args.cmd](args)


if __name__ == "__main__":
    main()
