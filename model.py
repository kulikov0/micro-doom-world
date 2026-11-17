import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEV, HUB_REPO, OUT

LATENT = 32  # channels per token
TOKENS = 48  # tokens per frame: a 6 x 8 grid, row-major
YAW_SCALE = 10.0  # degrees per frame -> action input
FWD, BACK, LEFT, RIGHT, RUN, ATTACK, USE, YAW = range(8)
T_CTX = 0.1  # noise level of the frames in the history
FPS = 17.5  # every second Doom tic
TURN_SPEED = 6.0  # degrees per frame while a turn key is held


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class Tokenizer(nn.Module):
    def __init__(self, latent=LATENT, enc=(64, 128, 192, 256), dec=(256, 192, 128, 64)):
        super().__init__()
        self.enc_in = nn.Conv2d(12, enc[0], 3, padding=1)
        self.enc_res = nn.ModuleList(Res(c) for c in enc)
        self.enc_down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1) for a, b in zip(enc, enc[1:]))
        self.enc_mid = Res(enc[-1])
        self.enc_out = nn.Conv2d(enc[-1], latent, 1)
        self.dec_in = nn.Conv2d(latent, dec[0], 3, padding=1)
        self.dec_res = nn.ModuleList(Res(c) for c in dec)
        self.dec_up = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(dec, dec[1:]))
        self.dec_out = nn.Conv2d(dec[-1], 12, 3, padding=1)

    def encode(self, x):
        h = self.enc_in(F.pixel_unshuffle(x, 2))
        for res, down in zip(self.enc_res, self.enc_down):
            h = down(res(h))
        return torch.tanh(self.enc_out(F.relu(self.enc_mid(self.enc_res[-1](h)))))

    def decode(self, z):
        h = self.dec_res[0](self.dec_in(z))
        for up, res in zip(self.dec_up, self.dec_res[1:]):
            h = res(up(F.interpolate(h, scale_factor=2, mode="nearest")))
        return F.pixel_shuffle(self.dec_out(F.relu(h)), 2)


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope_tables(frames, hd, device):
    fpos = torch.arange(frames, device=device) if isinstance(frames, int) else frames
    pos = fpos.float().repeat_interleave(TOKENS)
    cell = torch.arange(TOKENS, device=device).repeat(len(fpos))
    n_s = hd // 6
    n_t = hd // 2 - 2 * n_s

    def inv(n, base):
        return 1.0 / base ** (torch.arange(n, device=device).float() / n)
    ang = torch.cat([pos[:, None] * inv(n_t, 10000), (cell // 8).float()[:, None] * inv(n_s, 100),
                     (cell % 8).float()[:, None] * inv(n_s, 100)], -1)
    return ang.cos(), ang.sin()


def rope(x, cos, sin):
    a, b = x.chunk(2, -1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.qn, self.kn = nn.RMSNorm(self.hd), nn.RMSNorm(self.hd)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, cos, sin, mask, past=None):
        B, Fr, T, D = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c)[:, :, None].chunk(6, -1)
        h = (rms(x) * (1 + sc1) + sh1).reshape(B, Fr * T, D)
        q, k, v = self.qkv(h).view(B, Fr * T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), cos, sin), rope(self.kn(k), cos, sin)
        kk, vv = (k, v) if past is None else (torch.cat([past[0], k], 2), torch.cat([past[1], v], 2))
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        x = x + g1 * self.proj(o.transpose(1, 2).reshape(B, Fr, T, D))
        h = rms(x) * (1 + sc2) + sh2
        return x + g2 * self.fc2(F.gelu(self.fc1(h), approximate="tanh")), (k, v)


class WorldModel(nn.Module):
    def __init__(self, d=512, layers=8, heads=8, n_act=8, ch=LATENT):
        super().__init__()
        self.d, self.heads = d, heads
        self.inp = nn.Linear(ch, d)
        self.pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.a_mlp = nn.Sequential(nn.Linear(n_act, d), nn.SiLU(), nn.Linear(d, d))
        self.d_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.d_mlp[2].weight)
        nn.init.zeros_(self.d_mlp[2].bias)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out_ada = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, ch)
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def t_embed(self, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device) / 128)
        a = 1000 * t[..., None] * f
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t, a, pos=None, fmask=None, past=None, keep_kv=False, dstep=None, feats=False, rep=0):
        B, Fr = t.shape
        c = self.t_mlp(self.t_embed(t)) + self.a_mlp(a)
        c = F.silu(c + self.d_mlp(self.t_embed(torch.zeros_like(t) if dstep is None else dstep)))
        h = self.inp(x) + self.pos
        cos, sin = rope_tables(Fr if pos is None else pos, self.d // self.heads, x.device)
        if fmask is not None:
            mask = fmask.repeat_interleave(TOKENS, 0).repeat_interleave(TOKENS, 1)
        elif Fr == 1:
            mask = None
        else:
            n_past = 0 if past is None else past[0][0].shape[2] // TOKENS
            fr = torch.arange(Fr * TOKENS, device=x.device) // TOKENS
            mask = torch.cat([torch.ones(Fr * TOKENS, n_past * TOKENS, dtype=torch.bool, device=x.device),
                              fr[:, None] >= fr[None, :]], 1)
        kvs, hs = [], []
        for i, blk in enumerate(self.blocks):
            h, kv = blk(h, c, cos, sin, mask, None if past is None else past[i])
            kvs.append(kv)
            hs.append(h)
        if feats:
            return hs
        sh, sc = self.out_ada(c)[:, :, None].chunk(2, -1)
        out = self.out(rms(h) * (1 + sc) + sh)
        if rep:
            return out, hs[rep - 1]
        return (out, kvs) if keep_kv else out


def sampling_steps(args):
    return 1 if "adv" in args else 8


@torch.no_grad()
def generate(model, ctx, acts, horizon, steps=1, t_ctx=T_CTX, window=32, gen=None):
    B, C0 = ctx.shape[:2]
    dev = ctx.device
    dq = torch.full((B, 1), 1.0 / steps if steps < 8 else 0.0, device=dev)
    noisy = torch.stack([(1 - t_ctx) * ctx[:, i] + t_ctx * torch.randn(ctx[:, i].shape, device=dev, generator=gen)
                         for i in range(C0)], 1)
    s0 = max(0, C0 - (window - 1))
    _, cache = model(noisy[:, s0:], torch.full((B, C0 - s0), t_ctx, device=dev), acts[:, s0:C0],
                     pos=torch.arange(s0, C0, device=dev), keep_kv=True)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    out = []
    for k in range(horizon):
        p, a = torch.tensor([C0 + k], device=dev), acts[:, C0 + k:C0 + k + 1]
        x = torch.randn(B, TOKENS, ctx.shape[-1], device=dev, generator=gen)
        for s in range(steps):
            t, t_next = ts[s], ts[s + 1]
            x0 = model(x[:, None], torch.full((B, 1), float(t), device=dev), a, pos=p, past=cache, dstep=dq)[:, 0]
            x = x0 + (t_next / t) * (x - x0)
        out.append(x0)
        com = (1 - t_ctx) * x0 + t_ctx * torch.randn(x0.shape, device=dev, generator=gen)
        _, kv = model(com[:, None], torch.full((B, 1), t_ctx, device=dev), a, pos=p, past=cache, keep_kv=True)
        keep = (window - 2) * TOKENS
        cache = [(torch.cat([pk, nk], 2)[:, :, -keep - TOKENS:], torch.cat([pv, nv], 2)[:, :, -keep - TOKENS:])
                 for (pk, pv), (nk, nv) in zip(cache, kv)]
    return torch.stack(out, 1)


def to_x(frames):
    return torch.from_numpy(np.ascontiguousarray(frames)).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1


def to_u8(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def encode_frames(tok, frames, batch=512):
    return torch.cat([tok.encode(to_x(frames[i:i + batch])).half().cpu() for i in range(0, len(frames), batch)]).numpy()


@torch.no_grad()
def decode_latents(tok, lat, mean, std, batch=256):
    shape, ch = lat.shape[:-2], lat.shape[-1]
    mean, std = (torch.as_tensor(np.asarray(v, np.float32), device=lat.device) for v in (mean, std))
    z = (lat.reshape(-1, TOKENS, ch) * std + mean).transpose(1, 2).reshape(-1, ch, 6, 8)
    y = torch.cat([tok.decode(z[i:i + batch]) for i in range(0, len(z), batch)])
    return to_u8(y).reshape(*shape, 96, 128, 3)


def hub_file(path):
    local = os.environ.get("MICRO_DOOM_WEIGHTS")
    if local:
        return os.path.join(local, path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HUB_REPO, path)


def _local(name):
    return name if os.path.isabs(name) or os.path.exists(name) else os.path.join(OUT, name)


def load_tokenizer(name="tokenizer"):
    path = _local(name)
    if os.path.isfile(path):
        sd = torch.load(path, map_location=DEV)
    else:
        from safetensors.torch import load_file
        sd = load_file(hub_file("tokenizer/model.safetensors"), device=DEV)
    tok = Tokenizer(latent=sd["enc_out.weight"].shape[0]).to(DEV)
    tok.load_state_dict(sd)
    return tok.eval()


def load_world(name):
    path = _local(name)
    if os.path.isfile(path):
        ck = torch.load(path, map_location=DEV)
        sd, args, stats = ck["model"], ck["args"], ck["stats"]
    else:
        from safetensors.torch import load_file
        sd = load_file(hub_file(f"{name}/model.safetensors"), device=DEV)
        with open(hub_file(f"{name}/config.json")) as f:
            cfg = json.load(f)
        args, stats = cfg["args"], cfg["stats"]
    model = WorldModel(args["d"], args["layers"], args["heads"], ch=sd["inp.weight"].shape[1]).to(DEV)
    model.load_state_dict(sd)
    return model.eval(), args, (np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32))
