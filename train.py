"""
  python train.py --tag base
  python train.py --tag rf --init dyn_base.pt --rf 0.75 --repa 0 --batch 8 --lr 2e-4 --max-steps 5000 --schedule cosine
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, OUT, amp, load_dinov2, save
from model import TOKENS, YAW_SCALE, WorldModel, load_tokenizer, load_world

SHORTCUT_JUMPS = (0.25, 0.5, 1.0)


class Latents:
    def __init__(self, names, window, stats=None):
        ds = [np.load(os.path.join(OUT, f"lat_{n}.npz")) for n in names]
        raw = [d["latents"] for d in ds]
        self.ch, K = raw[0].shape[1], 65536
        if stats is None:
            s1, s2, n = 0.0, 0.0, 0
            for i in range(0, len(raw[0]), K):
                z = raw[0][i:i + K].astype(np.float64).reshape(-1, self.ch, TOKENS)
                s1, s2, n = s1 + z.sum((0, 2)), s2 + (z ** 2).sum((0, 2)), n + z.shape[0] * TOKENS
            stats = ((s1 / n).astype(np.float32), np.sqrt(s2 / n - (s1 / n) ** 2).astype(np.float32))
        self.mean, self.std = stats
        self.lat = torch.empty((sum(len(x) for x in raw), TOKENS, self.ch), dtype=torch.float16, device=DEV)
        i0 = 0
        for x in raw:
            for i in range(0, len(x), K):
                z = x[i:i + K].astype(np.float32).reshape(-1, self.ch, TOKENS).transpose(0, 2, 1)
                self.lat[i0 + i:i0 + i + len(z)] = torch.from_numpy(((z - self.mean) / self.std).astype(np.float16))
            i0 += len(x)
        eps, base = [], 0
        for d in ds:
            eps.append(d["episodes"] + base)
            base = eps[-1][-1] + 1
        ep, act = np.concatenate(eps), np.concatenate([d["actions"] for d in ds]).astype(np.float32)
        led = np.zeros_like(act)  # the action that led to frame i = the one taken at frame i - 1
        same = ep[1:] == ep[:-1]
        led[1:][same] = act[:-1][same]
        led[:, 7] /= YAW_SCALE
        self.act = torch.from_numpy(led).to(DEV)
        self.ep = ep
        self.starts = np.nonzero(ep[:len(ep) - window + 1] == ep[window - 1:])[0]  # windows inside one episode
        self.frames = [d["frames"] if "frames" in d else None for d in ds]

    def batch(self, rng, B, window):
        idx = torch.from_numpy(rng.choice(self.starts, B)[:, None] + np.arange(window)).to(DEV)
        return self.lat[idx].float(), self.act[idx]


def sample_t(shape, p_ctx, t_pow, t_ctx_max):
    ctx = torch.rand(shape, device=DEV) < p_ctx
    tgt = 1 - (1 - torch.rand(shape, device=DEV)) ** t_pow
    return torch.where(ctx, torch.rand(shape, device=DEV) * t_ctx_max, tgt)


def two_copy_mask(F_, G):
    m = torch.zeros(F_ + G, F_ + G, dtype=torch.bool, device=DEV)
    m[:F_, :F_] = torch.ones(F_, F_, dtype=torch.bool, device=DEV).tril()
    for g in range(G):
        m[F_ + g, :F_ - G + g] = True
        m[F_ + g, F_ + g] = True
    return m


def shortcut_inputs(teacher, x, a, t_ctx_max):
    B, F_ = x.shape[:2]
    D = torch.tensor(SHORTCUT_JUMPS, device=x.device)[torch.randint(len(SHORTCUT_JUMPS), (B,), device=x.device)]
    t1 = (torch.floor(torch.rand(B, device=x.device) / D) + 1) * D
    tc = torch.rand(B, F_ - 1, device=x.device) * t_ctx_max
    tt = torch.cat([tc, t1[:, None]], 1)
    xin = x * (1 - tt[..., None, None]) + torch.randn_like(x) * tt[..., None, None]
    h = D / 2
    hq = torch.where(h < 0.2, torch.zeros_like(h), h)  # half of a 1/4 jump is the plain flow model at step size 0

    def dv(v):
        return torch.cat([torch.zeros(B, F_ - 1, device=x.device), v[:, None]], 1)

    def ratio(u, v):
        return (u / v)[:, None, None]
    with torch.no_grad(), amp():
        xl = xin[:, -1]
        x0 = teacher(xin, tt, a, dstep=dv(hq))[:, -1].float()
        xm = x0 + ratio(t1 - h, t1) * (xl - x0)
        x0 = teacher(torch.cat([xin[:, :-1], xm[:, None]], 1), torch.cat([tc, (t1 - h)[:, None]], 1), a,
                     dstep=dv(hq))[:, -1].float()
        xe = x0 + ratio(t1 - D, t1 - h) * (xm - x0)
        k = ratio(t1 - D, t1)
        target = (xe - k * xl) / (1 - k)
    return xin, tt, dv(D), target


def rf_inputs(model, x, a, p_ctx, t_pow, s=0.6, t_ctx_max=0.2):
    B, F_ = x.shape[:2]
    dev = x.device
    ts = torch.sigmoid(torch.randn(B, 1, device=dev))
    ts = s * ts / (1 + (s - 1) * ts)
    tc = torch.rand(B, F_, device=dev) * t_ctx_max
    xs = x * (1 - ts[..., None, None]) + torch.randn_like(x) * ts[..., None, None]
    com, cache = torch.empty_like(x), None
    with torch.no_grad():
        x0 = model(xs[:, :1], ts, a[:, :1], pos=torch.zeros(1, dtype=torch.long, device=dev))[:, 0]
        for i in range(F_):
            com[:, i] = (1 - tc[:, i, None, None]) * x0 + tc[:, i, None, None] * torch.randn_like(x0)
            if i == F_ - 1:
                break
            out, kv = model(torch.stack([com[:, i], xs[:, i + 1]], 1), torch.cat([tc[:, i:i + 1], ts], 1),
                            a[:, i:i + 2], pos=torch.arange(i, i + 2, device=dev), past=cache, keep_kv=True)
            x0 = out[:, 1]
            kv = [(k[:, :, :TOKENS], v[:, :, :TOKENS]) for k, v in kv]
            cache = kv if cache is None else [(torch.cat([pk, k], 2), torch.cat([pv, v], 2))
                                              for (pk, pv), (k, v) in zip(cache, kv)]
    tg = sample_t((B, F_), p_ctx, t_pow, t_ctx_max)
    xin = torch.cat([com, x * (1 - tg[..., None, None]) + torch.randn_like(x) * tg[..., None, None]], 1)
    pos = torch.arange(F_, device=dev).repeat(2)
    return xin, torch.cat([tc, tg], 1), torch.cat([a, a], 1), pos, two_copy_mask(F_, F_)


class Repa:
    def __init__(self, model, enc, dec, mean, std, seed, gamma=0.7):
        self.rng = np.random.default_rng(seed + 1)
        self.enc = load_dinov2(enc).to(DEV).eval().requires_grad_(False)
        self.proj, self.gamma = nn.Conv2d(model.d, self.enc.embed_dim, 3, padding=1).to(DEV), gamma
        self.tok = load_tokenizer(dec).eval().requires_grad_(False)
        self.mean, self.std = torch.from_numpy(mean).to(DEV), torch.from_numpy(std).to(DEV)
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]

    @torch.no_grad()
    def target(self, x):
        z = (x * self.std + self.mean).transpose(1, 2).reshape(len(x), -1, 6, 8)
        with amp():
            img = (self.tok.decode(z).float().clamp(-1, 1) + 1) / 2
            img = F.interpolate(img, size=(84, 112), mode="bilinear", align_corners=False)
            img = (img - self.im_mean) / self.im_std
            f = self.enc.forward_features(img)["x_norm_patchtokens"].float()
        if self.gamma:
            f = f - self.gamma * f.mean(1, keepdim=True)
            f = f / (f.std(1, keepdim=True) + 1e-6)
        return f

    def loss(self, hid, x, n_frames):
        Fr = x.shape[1]
        idx = torch.from_numpy(self.rng.choice(Fr, n_frames, replace=False)).to(DEV)
        h = hid[:, -Fr:].index_select(1, idx).float().reshape(-1, TOKENS, hid.shape[-1])
        tgt = self.target(x.index_select(1, idx).reshape(-1, TOKENS, x.shape[-1]))
        z = self.proj(h.transpose(1, 2).reshape(len(h), -1, 6, 8)).flatten(2).transpose(1, 2)
        return 1 - F.cosine_similarity(z, tgt, dim=-1).mean()


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    data = Latents(args.data.split(","), args.frames)
    hold = Latents(args.hold.split(","), args.frames, (data.mean, data.std))
    stats = {"mean": data.mean.tolist(), "std": data.std.tolist()}
    if args.init:
        model, init_args, _ = load_world(args.init)
        args.d, args.layers, args.heads = init_args["d"], init_args["layers"], init_args["heads"]
    else:
        model = WorldModel(args.d, args.layers, args.heads, ch=data.ch).to(DEV)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95), fused=DEV == "cuda")
    ema_p, model_p = list(ema.parameters()), list(model.parameters())
    repa = None
    if args.repa:
        repa = Repa(model, args.repa_enc, args.repa_dec, data.mean, data.std, args.seed, args.repa_gamma)
        opt.add_param_group({"params": list(repa.proj.parameters()), "weight_decay": 0.0})
    fwd = torch.compile(model, mode=args.compile) if args.compile else model
    teacher = torch.compile(ema, mode=args.compile) if args.compile else ema
    print(f"{args.tag}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, window {args.frames}, batch "
          f"{args.batch}, {len(data.starts)} windows, {DEV}" + (f", from {args.init}" if args.init else "")
          + (f", resampling forcing {args.rf:.0%}" if args.rf else "") + f", shortcut {args.shortcut:.0%}"
          + (f", iREPA x{args.repa:g} after block {args.repa_layer}" if repa else ""), flush=True)

    vx, va = hold.batch(np.random.default_rng(123), 32, args.frames)
    g = torch.Generator(DEV).manual_seed(0)
    vt = torch.rand(vx.shape[:2], device=DEV, generator=g)
    veps = torch.randn(vx.shape, device=DEV, generator=g)
    ckpt, rpath = os.path.join(OUT, f"dyn_{args.tag}.pt"), os.path.join(OUT, f"dyn_{args.tag}_resume.pt")
    step, el0, next_eval, next_resume = 0, 0.0, 0.0, 0.0
    if os.path.exists(rpath):
        r = torch.load(rpath, map_location=DEV, weights_only=False)
        model.load_state_dict(r["model"])
        ema.load_state_dict(r["ema"])
        if repa:
            repa.proj.load_state_dict(r["repa"])
        opt.load_state_dict(r["opt"])
        step, el0, next_eval, next_resume = r["step"], r["el"], r["next_eval"], r["el"] + 60 * args.resume_min
        rng.bit_generator.state = r["rng"]
        torch.set_rng_state(r["torch_rng"].cpu())
        if DEV == "cuda":
            torch.cuda.set_rng_state(r["cuda_rng"].cpu())
        print(f"resumed from step {step}", flush=True)
    log_path = os.path.join(OUT, f"dyn_{args.tag}.jsonl")
    if not step and os.path.exists(log_path):
        os.remove(log_path)

    def predict(m, *inputs, **kw):
        out = m(*inputs, rep=args.repa_layer if repa else 0, **kw)
        return out if repa else (out, None)
    t0, loss_sum, n_loss = time.perf_counter() - el0, torch.zeros((), device=DEV), 0
    while True:
        el = time.perf_counter() - t0
        done = step >= args.max_steps
        if el >= next_eval or done:
            with torch.no_grad():
                vl = F.mse_loss(ema(vx * (1 - vt[..., None, None]) + veps * vt[..., None, None], vt, va), vx).item()
            rec = {"step": step, "min": el / 60, "loss": loss_sum.item() / max(n_loss, 1), "val_loss": vl}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:6d} {el / 60:6.1f} min  loss {rec['loss']:.4f}  held-out {vl:.4f}", flush=True)
            loss_sum, n_loss = torch.zeros((), device=DEV), 0
            next_eval += 60 * args.eval_min
            save({"model": ema.state_dict(), "args": vars(args), "stats": stats}, ckpt)
        if not done and el >= next_resume:
            save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                  "el": el, "next_eval": next_eval, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                  "cuda_rng": torch.cuda.get_rng_state() if DEV == "cuda" else None,
                  "repa": repa.proj.state_dict() if repa else None}, rpath)
            next_resume = el + 60 * args.resume_min
        if done:
            if os.path.exists(rpath):
                os.remove(rpath)
            break
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.max_steps)) if args.schedule == "cosine" else 1.0
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * decay
        for group in opt.param_groups:
            group["lr"] = lr
        x, a = data.batch(rng, args.batch, args.frames)
        u = rng.random()
        if u < args.rf:
            xin, tt, aa, pos, fm = rf_inputs(model, x, a, args.p_ctx, args.t_pow, args.rf_shift, args.t_ctx_max)
            with amp():
                out, hid = predict(model, xin, tt, aa, pos=pos, fmask=fm)
            mse = (out[:, -x.shape[1]:].float() - x).pow(2).mean()
        elif u < args.rf + args.shortcut:
            xin, tt, dv, target = shortcut_inputs(teacher, x, a, args.t_ctx_max)
            with amp():
                out, hid = predict(fwd, xin, tt, a, dstep=dv)
            mse = (out[:, -1].float() - target).pow(2).mean()
        else:
            t = sample_t(x.shape[:2], args.p_ctx, args.t_pow, args.t_ctx_max)
            xt = x * (1 - t[..., None, None]) + torch.randn_like(x) * t[..., None, None]
            with amp():
                out, hid = predict(fwd, xt, t, a)
            mse = (out.float() - x).pow(2).mean()
        loss = mse + args.repa * repa.loss(hid, x, args.repa_frames) if repa else mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, model_p, 1 - min(args.ema, (1 + step) / (10 + step)))
        loss_sum += mse.detach()
        n_loss += 1
        step += 1
        if args.keep_every and step % args.keep_every == 0:
            save({"model": ema.state_dict(), "args": vars(args), "stats": stats},
                 os.path.join(OUT, f"dyn_{args.tag}_{step}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--data", default="e1m1,e1m2,e1m3", help="latent files OUT/lat_<name>.npz")
    ap.add_argument("--hold", default="e1m1hold,e1m2hold,e1m3hold", help="held-out latents for the validation loss")
    ap.add_argument("--init", default="", help="start from the architecture and EMA weights of this checkpoint in OUT")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine"], help="cosine: down to 0.1 x lr")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--frames", type=int, default=32, help="window")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--p-ctx", type=float, default=0.3, help="share of frames at a history-like noise level")
    ap.add_argument("--t-pow", type=float, default=2.0, help="> 1 shifts the other frames towards pure noise")
    ap.add_argument("--t-ctx-max", type=float, default=0.2)
    ap.add_argument("--shortcut", type=float, default=0.25, help="share of shortcut bootstrap steps")
    ap.add_argument("--rf", type=float, default=0.0, help="share of resampling-forcing steps")
    ap.add_argument("--rf-shift", type=float, default=0.6)
    ap.add_argument("--repa", type=float, default=0.5, help="iREPA loss weight, 0 turns it off")
    ap.add_argument("--repa-layer", type=int, default=3)
    ap.add_argument("--repa-frames", type=int, default=8, help="frames per window aligned")
    ap.add_argument("--repa-enc", default="dinov2_vitb14_reg")
    ap.add_argument("--repa-dec", default="tokenizer_gan.pt", help="the decoder whose frames DINOv2 sees")
    ap.add_argument("--repa-gamma", type=float, default=0.7)
    ap.add_argument("--compile", default="", help="torch.compile mode on CUDA, e.g. default")
    ap.add_argument("--eval-min", type=float, default=5, help="validation loss and checkpoint every N minutes")
    ap.add_argument("--resume-min", type=float, default=2, help="full training state every N minutes")
    ap.add_argument("--keep-every", type=int, default=5000, help="also keep the EMA weights every N steps")
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
