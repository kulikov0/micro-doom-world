"""
  python play.py
  python play.py rf --start 9
  python play.py dyn_ladder.pt

Keys: W/S or Up/Down forward/back, A/D strafe, Left/Right turn, Shift run, Space or Ctrl fire, R new start, Esc quit.
"""
import argparse
import os
import time

import numpy as np
import torch

from common import DEV, OUT
from model import (ATTACK, BACK, FPS, FWD, LEFT, RIGHT, RUN, T_CTX, TOKENS, TURN_SPEED, YAW, YAW_SCALE, decode_latents,
                   hub_file, load_tokenizer, load_world, sampling_steps)


def load_starts(mean, std):
    path = os.path.join(OUT, "starts.npz")
    d = np.load(path if os.path.exists(path) else hub_file("starts.npz"))
    z = d["latents"].astype(np.float32)
    z = z.reshape(*z.shape[:3], TOKENS).transpose(0, 1, 3, 2)
    return torch.from_numpy((z - mean) / std).to(DEV), torch.from_numpy(d["actions"]).to(DEV), list(d["levels"])


class Session:
    @torch.no_grad()
    def __init__(self, model, ctx, acts, steps, window):
        self.m, self.steps, self.window = model, steps, window
        self.dq = torch.full((1, 1), 1.0 / steps if steps < 8 else 0.0, device=DEV)
        C0 = ctx.shape[1]
        noisy = (1 - T_CTX) * ctx + T_CTX * torch.randn_like(ctx)
        _, self.cache = model(noisy, torch.full((1, C0), T_CTX, device=DEV), acts, pos=torch.arange(C0, device=DEV),
                              keep_kv=True)
        self.ts = torch.linspace(1, 0, steps + 1, device=DEV)
        self.k, self.c0 = C0, C0

    @torch.no_grad()
    def step(self, a):
        p = torch.tensor([self.k], device=DEV)
        x = torch.randn(1, TOKENS, self.m.inp.in_features, device=DEV)
        for s in range(self.steps):
            t, t_next = self.ts[s], self.ts[s + 1]
            x0 = self.m(x[:, None], torch.full((1, 1), float(t), device=DEV), a, pos=p, past=self.cache,
                        dstep=self.dq)[:, 0]
            x = x0 + (t_next / t) * (x - x0)
        com = (1 - T_CTX) * x0 + T_CTX * torch.randn_like(x0)
        _, kv = self.m(com[:, None], torch.full((1, 1), T_CTX, device=DEV), a, pos=p, past=self.cache, keep_kv=True)
        keep = (self.window - 2) * TOKENS
        self.cache = [(torch.cat([pk, nk], 2)[:, :, -keep - TOKENS:], torch.cat([pv, nv], 2)[:, :, -keep - TOKENS:])
                      for (pk, pv), (nk, nv) in zip(self.cache, kv)]
        self.k += 1
        return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="ladder", help="ladder, rf or a checkpoint in OUT")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--start", type=int, default=-1, help="start 0-23, -1 for a random one")
    ap.add_argument("--yaw", type=float, default=TURN_SPEED, help="degrees per frame while an arrow key is held")
    ap.add_argument("--scale", type=int, default=5, help="window = 128 x 96 times this")
    args = ap.parse_args()
    model, margs, (mean, std) = load_world(args.model)
    steps = sampling_steps(margs)
    tok = load_tokenizer(args.tokenizer)
    lat, acts, levels = load_starts(mean, std)
    rng = np.random.default_rng()

    def new_session():
        i = args.start if args.start >= 0 else int(rng.integers(len(lat)))
        return i, Session(model, lat[i:i + 1], acts[i:i + 1], steps, margs["frames"])

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.init()
    W, H = 128 * args.scale, 96 * args.scale
    screen = pygame.display.set_mode((W, H))
    i, session = new_session()
    down = set()  # scancodes: the same keys on any keyboard layout

    def k(*codes):
        return float(any(c in down for c in codes))
    t_last, t_fps, n, running = time.perf_counter(), time.perf_counter(), 0, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.scancode == pygame.KSCAN_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                down.add(e.scancode)
                if e.scancode == pygame.KSCAN_R:
                    i, session = new_session()
            elif e.type == pygame.KEYUP:
                down.discard(e.scancode)
            elif e.type == pygame.WINDOWFOCUSLOST:
                down.clear()
        if not running:
            break
        a = torch.zeros(1, 1, 8, device=DEV)
        a[0, 0, FWD] = k(pygame.KSCAN_W, pygame.KSCAN_UP)
        a[0, 0, BACK] = k(pygame.KSCAN_S, pygame.KSCAN_DOWN)
        a[0, 0, LEFT] = k(pygame.KSCAN_A)
        a[0, 0, RIGHT] = k(pygame.KSCAN_D)
        a[0, 0, RUN] = k(pygame.KSCAN_LSHIFT, pygame.KSCAN_RSHIFT)
        a[0, 0, ATTACK] = k(pygame.KSCAN_SPACE, pygame.KSCAN_LCTRL, pygame.KSCAN_RCTRL)
        a[0, 0, YAW] = args.yaw * (k(pygame.KSCAN_LEFT) - k(pygame.KSCAN_RIGHT)) / YAW_SCALE  # + = left
        img = decode_latents(tok, session.step(a), mean, std)[0]
        screen.blit(pygame.transform.scale(pygame.surfarray.make_surface(img.transpose(1, 0, 2)), (W, H)), (0, 0))
        pygame.display.flip()
        wait = 1 / FPS - (time.perf_counter() - t_last)
        if wait > 0:
            time.sleep(wait)
        t_last = time.perf_counter()
        n += 1
        if t_last - t_fps > 1:
            pygame.display.set_caption(f"{args.model} | {levels[i]}, start {i} | frame {session.k - session.c0} | "
                                       f"{n / (t_last - t_fps):.0f} fps | R new start, Esc quit")
            t_fps, n = t_last, 0
    pygame.quit()


if __name__ == "__main__":
    main()
