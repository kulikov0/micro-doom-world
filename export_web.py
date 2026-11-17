"""
  python export_web.py ladder --out web/model --selftest 120 --check 3
"""
import argparse
import json
import math
import os
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LEVELS, OUT
from evaluate import CONTEXT, held_out, starts
from model import (ATTACK, BACK, FPS, FWD, LATENT, LEFT, RIGHT, RUN, T_CTX, TOKENS, TURN_SPEED, USE, YAW,
                   YAW_SCALE, load_tokenizer, load_world, rms)

SLOTS = 31


class Step(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m, self.H, self.hd = m, m.heads, m.d // m.heads
        n_s = self.hd // 6
        n_t = self.hd // 2 - 2 * n_s
        cell = torch.arange(TOKENS)
        inv_s = 1.0 / 100 ** (torch.arange(n_s).float() / n_s)
        self.register_buffer("inv_t", 1.0 / 10000 ** (torch.arange(n_t).float() / n_t))
        row, col = (cell // 8).float()[:, None], (cell % 8).float()[:, None]
        self.register_buffer("ang_s", torch.cat([row * inv_s, col * inv_s], -1))
        self.register_buffer("freq", torch.exp(-math.log(10000) * torch.arange(128) / 128))
        self.register_buffer("slot_frame", (torch.arange(SLOTS * TOKENS) // TOKENS).float())  # no int64 on WebGPU
        self.eps = torch.finfo(torch.float32).eps  # nn.RMSNorm(eps=None)

    def temb(self, t):
        a = 1000 * t[:, None] * self.freq
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t, a, dstep, pos, n_valid, cache_k, cache_v):
        m, H, hd = self.m, self.H, self.hd
        def mlp(seq, z):
            return seq[2](F.silu(seq[0](z)))
        c = F.silu(mlp(m.t_mlp, self.temb(t)) + mlp(m.a_mlp, a) + mlp(m.d_mlp, self.temb(dstep)))
        h = m.inp(x[0]) + m.pos
        ang = torch.cat([(pos * self.inv_t)[None].expand(TOKENS, -1), self.ang_s], -1)
        cos, sin = ang.cos(), ang.sin()
        mask = torch.cat([torch.where(self.slot_frame < n_valid, 0.0, -1e9), torch.zeros(TOKENS)])
        ks, vs = [], []
        for i, blk in enumerate(m.blocks):
            sh1, sc1, g1, sh2, sc2, g2 = blk.ada(c).chunk(6, -1)
            u = rms(h) * (1 + sc1) + sh1
            q, k, v = blk.qkv(u).view(TOKENS, 3, H, hd).permute(1, 2, 0, 3)

            def norm_rope(z, w):
                z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps) * w
                z1, z2 = z.chunk(2, -1)
                return torch.cat([z1 * cos - z2 * sin, z2 * cos + z1 * sin], -1)
            q, k = norm_rope(q, blk.qn.weight), norm_rope(k, blk.kn.weight)
            kk, vv = torch.cat([cache_k[i], k], 1), torch.cat([cache_v[i], v], 1)
            att = torch.softmax(q @ kk.transpose(1, 2) / math.sqrt(hd) + mask, -1)
            h = h + g1 * blk.proj((att @ vv).transpose(0, 1).reshape(TOKENS, H * hd))
            u = rms(h) * (1 + sc2) + sh2
            h = h + g2 * blk.fc2(F.gelu(blk.fc1(u), approximate="tanh"))
            ks.append(k)
            vs.append(v)
        sh, sc = m.out_ada(c).chunk(2, -1)
        # "+ 0 * x": x0 must come from a computing node, not a view of the last matmul: ORT copies a view into a
        # pre-allocated output outside the kernels, which a captured graph does not replay
        return m.out(rms(h) * (1 + sc) + sh)[None] + 0 * x, torch.stack(ks), torch.stack(vs)


class Decoder(nn.Module):
    def __init__(self, tok, mean, std):
        super().__init__()
        self.tok = tok
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))

    def forward(self, lat):
        # float 0-255, not rounded: ORT's WebGPU has no Round or uint8 Cast; the page rounds
        z = (lat[0] * self.std + self.mean).transpose(0, 1).reshape(1, -1, 6, 8)
        y = ((self.tok.decode(z).clamp(-1, 1) + 1) * 127.5)[0].permute(1, 2, 0)
        return torch.cat([y, torch.full((96, 128, 1), 255.0)], -1)


class Frame(nn.Module):
    def __init__(self, step, dec):
        super().__init__()
        self.step, self.dec = step, dec
        self.register_buffer("one", torch.ones(1))

    def forward(self, x, z, ctx, use_ctx, a, pos, n_valid, cache_k, cache_v):
        one = self.one
        x0, _, _ = self.step(x, one, a, one, pos, n_valid, cache_k, cache_v)
        src = use_ctx * ctx + (1 - use_ctx) * x0
        _, k, v = self.step((1 - T_CTX) * src + T_CTX * z, one * T_CTX, a, one * 0, pos, n_valid, cache_k, cache_v)
        return x0, self.dec(x0), k, v


INPUTS = ["x", "z", "ctx", "use_ctx", "a", "pos", "n_valid", "cache_k", "cache_v"]
OUTPUTS = ["x0", "rgba", "k_new", "v_new"]


class OrtWorld:
    def __init__(self, d):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(os.path.join(d, "frame.onnx"), so, providers=["CPUExecutionProvider"])
        meta = json.load(open(os.path.join(d, "meta.json")))
        self.shape = (meta["layers"], meta["heads"], SLOTS * TOKENS, meta["head_dim"])
        self.zero = np.zeros((1, TOKENS, meta["channels"]), np.float32)

    def run(self, x, z, ctx, use, a, pos):
        x0, _, kn, vn = self.sess.run(None, {"x": x, "z": z, "ctx": ctx, "use_ctx": np.float32([use]), "a": a,
                                             "pos": np.float32([pos]), "n_valid": np.float32([min(self.n, SLOTS)]),
                                             "cache_k": self.ck, "cache_v": self.cv})
        s = pos % SLOTS
        self.ck[:, :, s * TOKENS:(s + 1) * TOKENS], self.cv[:, :, s * TOKENS:(s + 1) * TOKENS] = kn, vn
        self.n += 1
        return x0

    def start(self, ctx, acts, noise):
        self.ck, self.cv, self.n = np.zeros(self.shape, np.float32), np.zeros(self.shape, np.float32), 0
        for i in range(len(ctx)):
            self.run(self.zero, noise[i][None], ctx[i][None], 1.0, acts[i][None], i)
        self.k = len(ctx)

    def frame(self, a, n0, n1):
        x0 = self.run(n0[None], n1[None], self.zero, 0.0, a[None], self.k)
        self.k += 1
        return x0[0]


@torch.no_grad()
def torch_world(model, ctx, acts, noise_ctx, noise):
    n_ctx = len(ctx)
    noisy = ((1 - T_CTX) * ctx + T_CTX * noise_ctx)[None]
    _, cache = model(noisy, torch.full((1, n_ctx), T_CTX), acts[None, :n_ctx], pos=torch.arange(n_ctx), keep_kv=True)
    out = []
    for j, a in enumerate(acts[n_ctx:]):
        p, a = torch.tensor([n_ctx + j]), a[None, None]
        x0 = model(noise[j, 0][None, None], torch.ones(1, 1), a, pos=p, past=cache, dstep=torch.ones(1, 1))[:, 0]
        com = (1 - T_CTX) * x0 + T_CTX * noise[j, 1][None]
        _, kv = model(com[:, None], torch.full((1, 1), T_CTX), a, pos=p, past=cache, keep_kv=True)
        keep = (SLOTS - 1) * TOKENS
        cache = [(torch.cat([pk, nk], 2)[:, :, -keep - TOKENS:], torch.cat([pv, nv], 2)[:, :, -keep - TOKENS:])
                 for (pk, pv), (nk, nv) in zip(cache, kv)]
        out.append(x0[0])
    return torch.stack(out)


def export_graph(model, tok, mean, std, path):
    L, H, hd, C = len(model.blocks), model.heads, model.d // model.heads, model.out.out_features
    cache = torch.zeros(L, H, SLOTS * TOKENS, hd)

    def lat():  # distinct tensors: the exporter merges aliased example inputs
        return torch.zeros(1, TOKENS, C)
    torch.onnx.export(Frame(Step(model), Decoder(tok, mean, std)).eval(),
                      (lat(), lat(), lat(), torch.zeros(1), torch.zeros(1, 8), torch.zeros(1), torch.zeros(1),
                       cache.clone(), cache.clone()), path,
                      input_names=INPUTS, output_names=OUTPUTS, dynamo=True, external_data=False, verbose=False)
    onnx.checker.check_model(path)


def write_starts(model_name, model, hold, st, out):
    ctx = np.stack([hold[f].lat[s:s + CONTEXT].float().cpu().numpy() for f, s in st])
    acts = np.stack([hold[f].act[s:s + CONTEXT].cpu().numpy() for f, s in st])
    with open(os.path.join(out, "starts.bin"), "wb") as fh:
        fh.write(ctx.tobytes())
        fh.write(acts.tobytes())
    raw = [np.load(os.path.join(OUT, f"lat_{lv.lower()}hold.npz"))["latents"] for lv in LEVELS]
    np.savez(os.path.join(OUT, "starts.npz"), latents=np.stack([raw[f][s:s + CONTEXT] for f, s in st]), actions=acts,
             levels=np.array([LEVELS[f] for f, _ in st]))
    meta = {"model": model_name, "context": CONTEXT, "slots": SLOTS, "layers": len(model.blocks), "heads": model.heads,
            "head_dim": model.d // model.heads, "tokens": TOKENS, "channels": ctx.shape[-1], "fps": FPS,
            "yaw_scale": YAW_SCALE, "yaw_deg_per_frame": TURN_SPEED,
            "actions": {"fwd": FWD, "back": BACK, "left": LEFT, "right": RIGHT, "run": RUN, "attack": ATTACK,
                        "use": USE, "yaw": YAW},
            "starts": [{"level": LEVELS[f], "offset": int(s)} for f, s in st],
            "starts_bin": {"latents": list(ctx.shape), "actions": list(acts.shape)}}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def noise(rng, n):
    return (torch.from_numpy(rng.standard_normal((CONTEXT, TOKENS, LATENT)).astype(np.float32)),
            torch.from_numpy(rng.standard_normal((n, 2, TOKENS, LATENT)).astype(np.float32)))


def write_selftest(model, hold, st, n, out):
    d = os.path.join(out, "selftest")
    os.makedirs(d, exist_ok=True)
    f, s = st[0]
    ctx, acts = hold[f].lat[s:s + CONTEXT].float().cpu(), hold[f].act[s:s + CONTEXT + n].cpu()
    nc, nz = noise(np.random.default_rng(1), n)
    ref = torch_world(model, ctx, acts, nc, nz)
    for name, arr in (("ctx_noise", nc), ("noise", nz), ("acts", acts[CONTEXT:]), ("ref", ref)):
        arr.numpy().tofile(os.path.join(d, f"{name}.bin"))
    with open(os.path.join(d, "info.json"), "w") as fh:
        json.dump({"start": 0, "frames": n}, fh)


def check(model, hold, st, n_starts, n, out):
    world, rng = OrtWorld(out), np.random.default_rng(0)
    for i in np.linspace(0, len(st) - 1, n_starts).round().astype(int):
        f, s = st[i]
        ctx, acts = hold[f].lat[s:s + CONTEXT].float().cpu(), hold[f].act[s:s + CONTEXT + n].cpu()
        nc, nz = noise(rng, n)
        ref = torch_world(model, ctx, acts, nc, nz).numpy()
        world.start(ctx.numpy(), acts[:CONTEXT].numpy(), nc.numpy())
        got = np.stack([world.frame(acts[CONTEXT + j].numpy(), nz[j, 0].numpy(), nz[j, 1].numpy()) for j in range(n)])
        diff = np.abs(got - ref).max((1, 2))
        print(f"start {i}: max |ONNX - PyTorch| frame 1 {diff[0]:.1e}, frame {n} {diff[-1]:.1e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="ladder, rf or a checkpoint in OUT")
    ap.add_argument("--out", default="web/model")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--selftest", type=int, default=120, help="frames of the page's ?selftest=1 reference")
    ap.add_argument("--check", type=int, default=0, help="parity of the exported graph vs PyTorch on N starts")
    ap.add_argument("--frames", type=int, default=240, help="frames per checked start")
    args = ap.parse_args()
    model, margs, (mean, std) = load_world(args.model)
    assert "adv" in margs, "the page makes a frame in one step: export a model after adversarial post-training"
    model, tok = model.cpu(), load_tokenizer(args.tokenizer).cpu()
    hold = held_out((mean, std))
    st = starts(hold)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    path = os.path.join(args.out, "frame.onnx")
    export_graph(model, tok, mean, std, path)
    write_starts(os.path.basename(args.model).removesuffix(".pt"), model, hold, st, args.out)
    print(f"{path}: {os.path.getsize(path) / 1e6:.0f} MB, exported in {time.perf_counter() - t0:.0f} s", flush=True)
    if args.selftest:
        write_selftest(model, hold, st, args.selftest, args.out)
    if args.check:
        check(model, hold, st, args.check, args.frames, args.out)


if __name__ == "__main__":
    main()
