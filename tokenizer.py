"""
  python tokenizer.py --tag tokenizer --steps 40500 --lr 3e-4
  python tokenizer.py --tag tokenizer_gan --init tokenizer.pt --adv 0.5 --adaptive --lr-d 4e-4 --d-steps 2 \
      --enemy-w 4 --enemy-share 0.5 --steps 9000
"""
import argparse
import json
import os
import time

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, DEV, OUT, amp
from model import Tokenizer, load_tokenizer, to_x


class PatchD(nn.Module):
    def __init__(self, ch=64, n=3):
        super().__init__()
        cs = [ch * min(2 ** i, 8) for i in range(n + 1)]
        layers = [nn.Conv2d(3, cs[0], 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
        for i in range(n):
            layers += [nn.Conv2d(cs[i], cs[i + 1], 4, stride=2 if i < n - 1 else 1, padding=1),
                       nn.GroupNorm(32, cs[i + 1]), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers, nn.Conv2d(cs[-1], 1, 4, padding=1))

    def forward(self, x):
        return self.net(x)


def enemy_pixels(packed):
    return np.concatenate([np.bitwise_count(packed[i:i + 65536]).reshape(-1, packed[0].size).sum(1)
                           for i in range(0, len(packed), 65536)])


def train(args):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    frames = np.load(os.path.join(DATA, "frames.npy"), mmap_mode="r")
    packed = np.load(os.path.join(DATA, "masks.npy"), mmap_mode="r")
    has_enemy = np.nonzero(enemy_pixels(packed) >= 30)[0]
    tok = load_tokenizer(args.init) if args.init else Tokenizer().to(DEV)
    tok.requires_grad_(False)
    trained = [p for n, p in tok.named_parameters() if n.startswith("dec_") or not args.init]
    for p in trained:
        p.requires_grad_(True)
    if args.adv:
        D = PatchD().to(DEV)
        optD = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9), weight_decay=0)
    perc = lpips.LPIPS(net="vgg", verbose=False).to(DEV).eval().requires_grad_(False)
    optG = torch.optim.AdamW(trained, lr=args.lr, betas=(0.5, 0.9), weight_decay=0)
    print(f"{args.tag}: {'decoder' if args.init else 'tokenizer'} {sum(p.numel() for p in trained) / 1e6:.1f}M "
          f"params, {len(frames)} frames, {len(has_enemy)} of them with an enemy, batch {args.batch}, GAN weight {args.adv}, "
          f"{DEV}", flush=True)
    log_path = os.path.join(OUT, f"tok_{args.tag}.jsonl")
    open(log_path, "w").close()
    t0, step, acc = time.perf_counter(), 0, []
    while step < args.steps:
        ne = int(args.batch * args.enemy_share)
        idx = np.concatenate([rng.choice(has_enemy, ne), rng.integers(0, len(frames), args.batch - ne)])
        x = to_x(frames[idx])
        m = torch.from_numpy(np.unpackbits(packed[idx], -1).astype(bool)).to(DEV).float()[:, None]
        m = F.max_pool2d(m, 5, 1, 2)  # a 2-px margin: the outline counts as the enemy
        with torch.set_grad_enabled(not args.init):
            z = tok.encode(x)
        with amp():
            y = tok.decode(z)
            w = 1 + args.enemy_w * m
            l1 = (w * (y - x).abs()).mean() / w.mean()
            lp = perc(y, x).mean()
            loss = l1 + args.lp * lp
            g_adv = torch.zeros((), device=DEV)
            if args.adv and step >= args.adv_start:
                logits = D(y)
                mw = 1 + args.enemy_w * F.adaptive_max_pool2d(m, logits.shape[-2:])
                g_adv = -(logits.float() * mw).mean() / mw.mean()
                lam = args.adv
                if args.adaptive:
                    last = tok.dec_out.weight
                    g_rec = torch.autograd.grad(loss, last, retain_graph=True)[0].norm()
                    g_gan = torch.autograd.grad(g_adv, last, retain_graph=True)[0].norm()
                    lam = args.adv * (g_rec / (g_gan + 1e-4)).clamp(0, 1e4).detach()
                loss = loss + lam * g_adv
        optG.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optG.step()
        d_loss = torch.zeros((), device=DEV)
        if args.adv:
            for _ in range(args.d_steps):
                with amp():
                    d_loss = F.relu(1 - D(x).float()).mean() + F.relu(1 + D(y.detach()).float()).mean()
                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()
        acc.append(torch.stack([l1.detach().float(), lp.detach().float(), g_adv.detach(), d_loss.detach()]))
        step += 1
        if step % 50 == 0:
            v = torch.stack(acc).mean(0).tolist()
            acc = []
            el = time.perf_counter() - t0
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "min": el / 60, "l1": v[0], "lpips": v[1], "g_adv": v[2],
                                    "d_loss": v[3]}) + "\n")
            if step % 500 == 0:
                print(f"step {step:6d} {el / 60:6.1f} min  l1 {v[0]:.4f}  lpips {v[1]:.4f}  g_adv {v[2]:+.3f}  "
                      f"d_loss {v[3]:.3f}", flush=True)
            if step % 5000 == 0:
                torch.save(tok.state_dict(), os.path.join(OUT, f"{args.tag}.pt"))
    torch.save(tok.state_dict(), os.path.join(OUT, f"{args.tag}.pt"))
    print(f"{step} steps in {(time.perf_counter() - t0) / 60:.1f} min -> {args.tag}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tokenizer")
    ap.add_argument("--init", default="", help="fine-tune only the decoder of this tokenizer in OUT")
    ap.add_argument("--steps", type=int, default=40500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--lp", type=float, default=0.1, help="LPIPS-VGG weight")
    ap.add_argument("--adv", type=float, default=0.0, help="GAN weight, 0 trains without a discriminator")
    ap.add_argument("--adv-start", type=int, default=500, help="discriminator-only steps first")
    ap.add_argument("--d-steps", type=int, default=1, help="discriminator updates per decoder update")
    ap.add_argument("--adaptive", action="store_true", help="VQGAN adaptive GAN weight")
    ap.add_argument("--enemy-w", type=float, default=0.0, help="extra weight on enemy pixels and patches")
    ap.add_argument("--enemy-share", type=float, default=0.0, help="share of each batch from frames with an enemy")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
