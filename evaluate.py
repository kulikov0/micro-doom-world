"""
  python evaluate.py ladder rf dyn_base.pt
"""
import argparse
import functools
import glob
import os

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, DEV, LEVELS, OUT, load_dinov2
from model import decode_latents, generate, load_tokenizer, load_world, sampling_steps, to_x
from train import Latents

CONTEXT, HORIZON, WINDOW = 8, 480, 32
HORIZONS = (1, 5, 15, 60, 240, 480)
KD_WINDOWS = (0, 14)
CRITIC_WINDOWS = (0, 7, 14)
SEEDS = 3


def held_out(stats):
    return [Latents([f"{lv.lower()}hold"], 2, stats) for lv in LEVELS]


def starts(hold):
    out = []
    for f, h in enumerate(hold):
        seg = [e for e in np.unique(h.ep) if (h.ep == e).sum() >= 30 + CONTEXT + HORIZON]
        out += [(f, int(np.nonzero(h.ep == seg[k])[0][0]) + 30) for k in np.linspace(0, len(seg) - 1, 8).astype(int)]
    return out


def rollouts(name, model, args, hold, st):
    path = os.path.join(OUT, f"eval_{name}.npz")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)["lat"]).to(DEV).float()
    idx = [torch.arange(s, s + CONTEXT + HORIZON, device=DEV) for _, s in st]
    lat = torch.stack([hold[f].lat[i] for (f, _), i in zip(st, idx)]).float()
    act = torch.stack([hold[f].act[i] for (f, _), i in zip(st, idx)])
    out = [generate(model, lat[b:b + 12, :CONTEXT], act[b:b + 12], HORIZON, steps=sampling_steps(args),
                    window=args["frames"], gen=torch.Generator(DEV).manual_seed(b)).half().cpu()
           for b in range(0, len(st), 12)]
    np.savez(path, lat=torch.cat(out).numpy())
    return torch.cat(out).to(DEV).float()


@torch.no_grad()
def lpips_table(lp, frames, real):
    return np.stack([lp(to_x(frames[:, h - 1]), to_x(real[:, h - 1])).flatten().cpu().numpy() for h in HORIZONS], 1)


@functools.cache
def dino_model():
    return load_dinov2("dinov2_vitl14").to(DEV).eval()


@torch.no_grad()
def dino(frames):
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    out = []
    for i in range(0, len(frames), 128):
        x = torch.from_numpy(np.ascontiguousarray(frames[i:i + 128])).to(DEV).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        out.append(dino_model()((x - mean) / std).float().cpu())
    return torch.cat(out)


def reference_features():
    path = os.path.join(OUT, "eval_reference.npy")
    if not os.path.exists(path):
        rng, feats = np.random.default_rng(0), []
        for lv in LEVELS:
            paths = sorted(glob.glob(os.path.join(DATA, f"{lv.lower()}_*.npz")))
            sizes = [len(np.load(p)["episodes"]) for p in paths]
            pick = np.sort(rng.choice(sum(sizes), 2000, replace=False))
            frames, start = [], 0
            for p, n in zip(paths, sizes):
                sel = pick[(pick >= start) & (pick < start + n)] - start
                if len(sel):
                    frames.append(np.load(p)["frames"][sel])
                start += n
            feats.append(dino(np.concatenate(frames)))
        np.save(path, torch.cat(feats).numpy())
    return torch.from_numpy(np.load(path))


class KernelDistance:
    def __init__(self, y):
        self.y, self.d = y.double(), y.shape[1]
        k = self.k(self.y, self.y)
        n = len(y)
        self.kyy = float((k.sum() - k.diagonal().sum()) / (n * (n - 1)))

    def k(self, a, b):
        return (a @ b.T / self.d + 1) ** 3

    def __call__(self, x):
        x, m = x.double(), len(x)
        kxx = self.k(x, x)
        return float((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1)) + self.kyy - 2 * self.k(x, self.y).mean())


def jackknife(fn, x, groups):
    ids = np.unique(groups)
    loo = np.array([fn(x[torch.from_numpy(groups != g)]) for g in ids])
    return fn(x), float(np.sqrt((len(ids) - 1) / len(ids) * ((loo - loo.mean()) ** 2).sum()))


def critic_net():
    ch, layers = (6, 32, 64, 96, 96), []
    for a, b in zip(ch, ch[1:]):
        layers += [nn.Conv2d(a, b, 4, 2, 1), nn.LeakyReLU(0.2)]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch[-1], 1)).to(DEV)


def critic_inputs(now, prev):
    x, p = (torch.from_numpy(np.ascontiguousarray(a)).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1
            for a in (now, prev))
    return torch.cat([x, 2 * (x - p)], 1)


def train_critic(fakes, real, starts_idx, steps, seed, bs=64):
    torch.manual_seed(seed)
    rng, m = np.random.default_rng(seed), critic_net()
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    pairs = np.array([(i, t) for i in starts_idx for t in range(1, HORIZON)])
    names, h = list(fakes), bs // 2
    for _ in range(steps):
        k, v = rng.integers(len(pairs), size=bs), rng.integers(len(names), size=h)
        s, t = pairs[k, 0], pairs[k, 1]
        now = np.concatenate([np.stack([fakes[names[v[i]]][s[i], t[i]] for i in range(h)]), real[s[h:], t[h:]]])
        prev = np.concatenate([np.stack([fakes[names[v[i]]][s[i], t[i] - 1] for i in range(h)]),
                               real[s[h:], t[h:] - 1]])
        x = critic_inputs(now, prev)
        x = torch.where(torch.rand(bs, 1, 1, 1, device=DEV) < 0.5, x.flip(-1), x)
        y = torch.cat([torch.zeros(h), torch.ones(bs - h)]).to(DEV)
        loss = F.binary_cross_entropy_with_logits(m(x)[:, 0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


@torch.no_grad()
def p_real(m, video, idx, w):
    t = np.arange(max(1, w * WINDOW), (w + 1) * WINDOW)
    sub = video[idx]
    now, prev = sub[:, t].reshape(-1, *video.shape[2:]), sub[:, t - 1].reshape(-1, *video.shape[2:])
    p = torch.cat([torch.sigmoid(m(critic_inputs(now[i:i + 512], prev[i:i + 512]))[:, 0]).cpu()
                   for i in range(0, len(now), 512)]).numpy()
    return p.reshape(len(idx), len(t)).mean(1)


def critic_scores(fakes, real, n, steps):
    rows = {name: {w: np.zeros((SEEDS, n)) for w in CRITIC_WINDOWS} for name in ["real game", *fakes]}
    halves = (np.r_[0:n:2], np.r_[1:n:2])
    for fold, (tr, te) in enumerate((halves, halves[::-1])):
        for sd in range(SEEDS):
            m = train_critic(fakes, real, tr, steps, 10 * fold + sd)
            for w in CRITIC_WINDOWS:
                rows["real game"][w][sd, te] = p_real(m, real, te, w)
                for name, video in fakes.items():
                    rows[name][w][sd, te] = p_real(m, video, te, w)
    return rows


def frames_label(windows):
    return " / ".join(f"{w * WINDOW + 1}-{(w + 1) * WINDOW}" for w in windows)


def report_lpips(fakes, real):
    lp = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    print("\nLPIPS to the real frame at " + " / ".join(map(str, HORIZONS)) + "; lower is better, +- s.e. over starts")
    for name, frames in fakes.items():
        v = lpips_table(lp, frames, real)
        print(f"  {name:24s} " + "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(v.mean(0), v.std(0) / np.sqrt(len(v)))))


def report_kd(fakes, real):
    kd = KernelDistance(reference_features())
    groups = np.repeat(np.arange(len(real)), WINDOW)
    print(f"\nKernel distance x1e3 on DINOv2 features, frames {frames_label(KD_WINDOWS)}"
          "; lower is better, +- jackknife s.e.")
    for name, frames in {"real game": real, **fakes}.items():
        cells = []
        for w in KD_WINDOWS:
            win = frames[:, w * WINDOW:(w + 1) * WINDOW]
            v, se = jackknife(kd, dino(win.reshape(-1, *win.shape[2:])), groups)
            cells.append(f"{v * 1e3:7.1f}+-{se * 1e3:5.1f}")
        print(f"  {name:24s} " + "  ".join(cells), flush=True)


def report_critic(fakes, real_tok, steps):
    rows = critic_scores(fakes, real_tok, len(real_tok), steps)
    print(f"\nCritic P(real), frames {frames_label(CRITIC_WINDOWS)}"
          "; real frames pass through the same tokenizer; higher is better, +- s.e. over starts")
    for name, r in rows.items():
        per_start = [r[w].mean(0) for w in CRITIC_WINDOWS]
        print(f"  {name:24s} " + "  ".join(f"{p.mean():.2f}+-{p.std() / np.sqrt(len(p)):.2f}" for p in per_start))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="ladder, rf or checkpoints in OUT")
    ap.add_argument("--tokenizer", default="tokenizer", help="decoder for the generated latents")
    ap.add_argument("--critic-steps", type=int, default=1500)
    args = ap.parse_args()
    _, _, stats = load_world(args.models[0])
    hold = held_out(stats)
    st = starts(hold)
    tok = load_tokenizer(args.tokenizer)
    real = np.stack([hold[f].frames[0][s + CONTEXT:s + CONTEXT + HORIZON] for f, s in st])
    fakes = {}
    for name in args.models:
        model, margs, _ = load_world(name)
        key = os.path.basename(name).removesuffix(".pt")
        fakes[key] = decode_latents(tok, rollouts(key, model, margs, hold, st), *stats)
    report_lpips(fakes, real)
    report_kd(fakes, real)
    lat = torch.stack([hold[f].lat[s + CONTEXT:s + CONTEXT + HORIZON] for f, s in st]).float()
    report_critic(fakes, decode_latents(tok, lat, *stats), args.critic_steps)


if __name__ == "__main__":
    main()
