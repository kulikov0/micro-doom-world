"""
  python adversarial.py --init dyn_base.pt --tag ladder --max-steps 750 --segs 2 --batch 8 --accum 2
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, OUT, amp, save
from model import TOKENS, generate, load_world, rms
from train import Latents


class Disc(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        L, d = len(body.blocks), body.d
        self.taps = (L // 2 - 1, L - 1)
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1)) for _ in self.taps)
        for head in self.heads:
            nn.init.zeros_(head[2].weight)
            nn.init.zeros_(head[2].bias)

    def forward(self, x, t, a):
        hs = self.body(x, t, a, feats=True)
        return sum(head(rms(hs[i])).squeeze(-1).float().mean(-1) for i, head in zip(self.taps, self.heads))

    def params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("body.out", "body.out_ada"))]


def encode(G, ctx, a, t_ctx):
    B, C0 = ctx.shape[:2]
    noisy = (1 - t_ctx) * ctx + t_ctx * torch.randn_like(ctx)
    with amp():
        _, cache = G(noisy, torch.full((B, C0), t_ctx, device=DEV), a, pos=torch.arange(C0, device=DEV), keep_kv=True)
    return cache


def student(G, cache, pending, a, p0, n, K, t_ctx, window):
    B = a.shape[0]
    ts = torch.linspace(1, 0, K + 1, device=DEV)
    dq = torch.full((B, 1), 1.0 / K if K < 8 else 0.0, device=DEV)
    out = []
    for i in range(n):
        p = p0 + i
        z = torch.randn(B, TOKENS, G.inp.in_features, device=DEV)
        for s in range(K):
            t = torch.full((B, 1), float(ts[s]), device=DEV)
            with amp():
                if s == 0 and pending is not None:
                    keep = (window - 2) * TOKENS
                    cache = [(ck[:, :, -keep:], cv[:, :, -keep:]) for ck, cv in cache]
                    o, kv = G(torch.stack([pending[0], z], 1), torch.cat([torch.full_like(t, t_ctx), t], 1),
                              torch.cat([pending[1], a[:, i:i + 1]], 1), pos=torch.arange(p - 1, p + 1, device=DEV),
                              past=cache, keep_kv=True, dstep=torch.cat([torch.zeros_like(dq), dq], 1))
                    cache = [(torch.cat([ck, nk[:, :, :TOKENS]], 2), torch.cat([cv, nv[:, :, :TOKENS]], 2))
                             for (ck, cv), (nk, nv) in zip(cache, kv)]
                    x0 = o[:, 1].float()
                else:
                    x0 = G(z[:, None], t, a[:, i:i + 1], pos=torch.tensor([p], device=DEV), past=cache,
                           dstep=dq)[:, 0].float()
            z = x0 + (ts[s + 1] / ts[s]) * (z - x0)
        out.append(x0)
        pending = ((1 - t_ctx) * x0.detach() + t_ctx * torch.randn_like(x0), a[:, i:i + 1])
    return torch.stack(out, 1), cache, pending


def d_logits(D, frames, a, td, eps, C0):
    t = td[:, None].expand(frames.shape[:2])
    with amp():
        return D(frames * (1 - t[..., None, None]) + eps * t[..., None, None], t, a)[:, C0:]


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    G, ga, (mean, std) = load_world(args.init)
    window, C0, n, S, K = ga["frames"], args.ctx, args.seg, args.segs, args.gen_steps
    assert C0 <= n and C0 + n <= window and args.batch % args.accum == 0
    data = Latents(args.data.split(","), C0 + S * n, (mean, std))
    hold = Latents(args.hold.split(","), C0 + args.mon_frames, (mean, std))
    G.train().requires_grad_(True)
    ema = copy.deepcopy(G).eval().requires_grad_(False)
    D = Disc(load_world(args.init)[0]).to(DEV).train()
    optG = torch.optim.RMSprop(G.parameters(), lr=args.lr, alpha=0.9)
    optD = torch.optim.RMSprop(D.params(), lr=args.lr_d, alpha=0.9)
    ema_p, g_p = list(ema.parameters()), list(G.parameters())
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    ck_args = dict(ga, adv=vars(args))
    print(f"{args.tag}: from {args.init}, {C0} real frames then {S} x {n} own frames, batch {args.batch}, "
          f"lr G {args.lr:g} / D {args.lr_d:g}, {DEV}", flush=True)
    vx, va = hold.batch(np.random.default_rng(123), 32, C0 + args.mon_frames)
    ckpt, rpath = os.path.join(OUT, f"dyn_{args.tag}.pt"), os.path.join(OUT, f"dyn_{args.tag}_resume.pt")
    step, el0, next_eval, next_resume = 0, 0.0, 0.0, 0.0
    if os.path.exists(rpath):
        r = torch.load(rpath, map_location=DEV, weights_only=False)
        for m, k in ((G, "G"), (ema, "ema"), (D, "D"), (optG, "optG"), (optD, "optD")):
            m.load_state_dict(r[k])
        step, el0, next_eval, next_resume = r["step"], r["el"], r["next_eval"], r["el"] + 60 * args.resume_min
        rng.bit_generator.state = r["rng"]
        torch.set_rng_state(r["torch_rng"].cpu())
        print(f"resumed from step {step}", flush=True)
    log_path = os.path.join(OUT, f"dyn_{args.tag}.jsonl")
    if not step and os.path.exists(log_path):
        os.remove(log_path)

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    edges = [e for e in (0, 8, 24, 48) if e < args.mon_frames] + [args.mon_frames]
    t0, stopped = time.perf_counter() - el0, False
    sums, n_sums = torch.zeros(4, device=DEV), 0
    while True:
        el = time.perf_counter() - t0
        done = step >= args.max_steps
        if el >= next_eval or done:
            gen = generate(ema, vx[:, :C0], va, args.mon_frames, steps=K, window=window,
                           gen=torch.Generator(DEV).manual_seed(0))
            eg, er = gen.pow(2).mean((2, 3)), vx[:, C0:].pow(2).mean((2, 3))
            energy = {f"{a + 1}-{b}": (eg[:, a:b].mean() / er[:, a:b].mean()).item() for a, b in zip(edges, edges[1:])}
            log({"step": step, "min": el / 60, "energy": energy})
            print(f"step {step:6d} {el / 60:6.1f} min  energy gen / real "
                  + " ".join(f"{k} {v:.2f}" for k, v in energy.items()), flush=True)
            save({"model": ema.state_dict(), "args": ck_args, "stats": stats}, ckpt)
            next_eval += 60 * args.eval_min
            stopped = args.stop_energy and max(energy.values()) > args.stop_energy
            done = done or stopped
        if el >= next_resume or done:  # kept at the end: the next ladder stage continues from it
            save({"G": G.state_dict(), "ema": ema.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(),
                  "optD": optD.state_dict(), "step": step, "el": el, "next_eval": next_eval,
                  "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, rpath)
            next_resume = el + 60 * args.resume_min
        if done:
            if stopped:
                raise SystemExit(f"stopped at step {step}: frame energy above {args.stop_energy:g} x the real one")
            break
        warm = min(1.0, (step + 1) / args.warmup)
        g_on = step >= args.d_warmup
        for group in optG.param_groups:
            group["lr"] = args.lr * warm
        for group in optD.param_groups:
            group["lr"] = args.lr_d * warm
        xb, ab = data.batch(rng, args.batch, C0 + S * n)
        mb = args.batch // args.accum
        chunks = [(xb[i:i + mb], ab[i:i + mb]) for i in range(0, args.batch, mb)]
        W = S * len(chunks)

        D.requires_grad_(False)
        optG.zero_grad(set_to_none=True)
        all_segs, g_loss = [], torch.zeros((), device=DEV)
        for x, a in chunks:
            B, segs = x.shape[0], []
            with torch.set_grad_enabled(g_on):
                cache, pending = encode(G, x[:, :C0], a[:, :C0], args.t_ctx), None
                for j in range(S):
                    p0 = C0 + j * n
                    gen, cache, pending = student(G, cache, pending, a[:, p0:p0 + n], p0, n, K, args.t_ctx, window)
                    if g_on:
                        ctx = x[:, :C0] if j == 0 else segs[-1][:, -C0:]
                        td = torch.rand(B, device=DEV) * args.d_noise
                        eps = torch.randn(B, C0 + n, *x.shape[2:], device=DEV)
                        aw = a[:, p0 - C0:p0 + n]
                        with torch.no_grad():
                            lreal = d_logits(D, x[:, p0 - C0:p0 + n], aw, td, eps, C0)
                        lfake = d_logits(D, torch.cat([ctx, gen], 1), aw, td, eps, C0)
                        loss = F.softplus(lreal - lfake).mean() / W
                        loss.backward()
                        g_loss += loss.detach()
                    segs.append(gen.detach())
                    cache = [(ck.detach(), cv.detach()) for ck, cv in cache]
            all_segs.append(segs)
            del cache, pending, gen
        if g_on:
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            optG.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema_p, g_p, 1 - min(args.ema, (1 + step) / (10 + step)))

        D.requires_grad_(True)
        optD.zero_grad(set_to_none=True)
        st = torch.zeros(3, device=DEV)
        for (x, a), segs in zip(chunks, all_segs):
            for j in range(S):
                p0 = C0 + j * n
                ctx = x[:, :C0] if j == 0 else segs[j - 1][:, -C0:]
                real, fake, aw = x[:, p0 - C0:p0 + n], torch.cat([ctx, segs[j]], 1), a[:, p0 - C0:p0 + n]
                td, eps = torch.rand(x.shape[0], device=DEV) * args.d_noise, torch.randn_like(real)
                lreal, lfake = d_logits(D, real, aw, td, eps, C0), d_logits(D, fake, aw, td, eps, C0)
                r1 = (lreal - d_logits(D, real + args.r1_sigma * torch.randn_like(real), aw, td, eps, C0)).pow(2).mean()
                r2 = (lfake - d_logits(D, fake + args.r1_sigma * torch.randn_like(fake), aw, td, eps, C0)).pow(2).mean()
                d_loss = F.softplus(lfake - lreal).mean()
                ((d_loss + args.r1 * (r1 + r2)) / W).backward()
                st += torch.stack([d_loss, (lreal > lfake).float().mean(), (lreal - lfake).mean()]).detach() / W
        torch.nn.utils.clip_grad_norm_(D.params(), 1.0)
        optD.step()
        sums += torch.cat([g_loss[None], st])
        n_sums += 1
        if DEV == "mps":
            torch.mps.empty_cache()  # the MPS cache otherwise grows past the machine's memory
        step += 1
        if step % 10 == 0:
            v = (sums / n_sums).tolist()
            sums, n_sums = torch.zeros(4, device=DEV), 0
            log({"step": step, "min": el / 60, "g_loss": v[0], "d_loss": v[1], "d_acc": v[2], "logit_gap": v[3]})
            if step % 50 == 0:
                print(f"step {step:6d} {el / 60:6.1f} min  G {v[0]:.3f}  D {v[1]:.3f}  D acc {v[2]:.2f}  "
                      f"gap {v[3]:+.2f}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="checkpoint in OUT to start from: a base with shortcut training")
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--data", default="e1m1,e1m2,e1m3")
    ap.add_argument("--hold", default="e1m1hold,e1m2hold,e1m3hold", help="held-out latents for the energy monitor")
    ap.add_argument("--max-steps", type=int, default=750)
    ap.add_argument("--segs", type=int, default=2, help="segments of --seg own frames in a row")
    ap.add_argument("--seg", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches one after another")
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--lr-d", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--d-warmup", type=int, default=250, help="discriminator-only steps first")
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--gen-steps", type=int, default=1, help="jumps per generated frame")
    ap.add_argument("--ctx", type=int, default=8, help="real context frames")
    ap.add_argument("--t-ctx", type=float, default=0.1)
    ap.add_argument("--d-noise", type=float, default=0.2, help="discriminator inputs noised to U(0, this)")
    ap.add_argument("--r1", type=float, default=1000.0, help="approximate R1 / R2 weight")
    ap.add_argument("--r1-sigma", type=float, default=0.1)
    ap.add_argument("--mon-frames", type=int, default=192, help="energy monitor rollout length")
    ap.add_argument("--stop-energy", type=float, default=1.5, help="stop when a monitor energy bin exceeds this")
    ap.add_argument("--eval-min", type=float, default=10)
    ap.add_argument("--resume-min", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
