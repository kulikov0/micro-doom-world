import contextlib
import os

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MICRO_DOOM_OUT", os.path.join(ROOT, "out"))
DATA = os.path.join(OUT, "data")
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LEVELS = ("E1M1", "E1M2", "E1M3")
HUB_REPO = "kulikov0/micro-doom-world"


def amp():
    if DEV == "cuda":
        return torch.autocast("cuda", torch.bfloat16)
    mode = os.environ.get("MICRO_DOOM_MPS_AMP")
    if DEV == "mps" and mode:
        return torch.autocast("mps", {"bf16": torch.bfloat16, "fp16": torch.float16}[mode])
    return contextlib.nullcontext()


def load_dinov2(name):
    local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(local):
        return torch.hub.load(local, name, source="local")
    return torch.hub.load("facebookresearch/dinov2", name, verbose=False, trust_repo=True)


def save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)

