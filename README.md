# micro-doom-world

A 39M-parameter world model of a Doom-like game. Given the previous frames and the player's keys, it generates the
next 128x96 frame. Play it in the browser:
[huggingface.co/spaces/kulikov0/micro-doom-world](https://huggingface.co/spaces/kulikov0/micro-doom-world).

https://github.com/user-attachments/assets/01432868-9f3d-4941-a10e-9fcfba22bf74

The ladder model played with the keyboard. Every frame of the video is generated.

The weights, the architecture and the evaluation are on the model card:
[huggingface.co/kulikov0/micro-doom-world](https://huggingface.co/kulikov0/micro-doom-world).

## Play locally

With Python 3.13:

    pip install -r requirements.txt
    python play.py
    python play.py rf

`play.py` downloads the weights from the model repository and opens a window. Keys: W/S or Up/Down forward and back,
A/D strafe, Left/Right turn, Shift run, Space or Ctrl fire, R new start, Esc quit.

## Train

Run the scripts from the repository root, in order. Everything is written to `out/`, or to `$MICRO_DOOM_OUT` if it is
set.

| Script | What it does |
|---|---|
| `scripts/1_data.sh` | records a scripted player on E1M1-E1M3: 500k training frames and 2 held-out games per level |
| `scripts/2_tokenizer.sh` | trains the tokenizer and its GAN decoder, then encodes every frame |
| `scripts/3_base.sh` | trains the base model, 30k steps at batch 32 |
| `scripts/4_rf.sh` | resampling forcing on top of the base, 5k steps |
| `scripts/5_ladder.sh` | adversarial post-training in three stages: 750, 1250 and 1750 steps |
| `scripts/6_evaluate.sh` | LPIPS, kernel distance and the critic on 24 held-out starts |
| `scripts/7_export.sh` | ONNX graphs and start files for the browser page |

The `ladder` model is scripts 1, 2, 3 and 5. The `rf` model adds script 4 before the post-training:

    scripts/4_rf.sh
    scripts/5_ladder.sh dyn_rf.pt rf_ladder

`scripts/6_evaluate.sh` compares your `dyn_ladder_1750.pt` and `dyn_base.pt` with the published `ladder` and `rf`.

The data take about 90 GB of disk. 55 GB of it is `data/frames.npy`, which only the tokenizer training reads; it can be
deleted after `scripts/2_tokenizer.sh`.

## Browser page

`web/` is the page of the Space. After `scripts/7_export.sh`:

    python -m http.server -d web 8000

Open http://localhost:8000 in a browser with WebGPU. `?model=rf` switches the model, `?selftest=1` compares the page
with PyTorch.

## Files

| File | Contents |
|---|---|
| `model.py` | tokenizer, dynamics model, generation with a KV cache, weight loading |
| `train.py` | base training and the resampling forcing stage |
| `adversarial.py` | adversarial post-training |
| `tokenizer.py` | tokenizer and GAN decoder training |
| `game.py`, `bot.py` | ViZDoom setup, level geometry from the WAD, the scripted player |
| `data.py` | recording, audit, unpacking and encoding of the data |
| `evaluate.py` | LPIPS, kernel distance and the critic |
| `play.py` | local play |
| `export_web.py` | ONNX export for the page |
| `common.py` | paths and device |

## License

MIT, see `LICENSE`. The levels come from [Freedoom](https://freedoom.github.io/) and are rendered with
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom). Freedoom is BSD-3-Clause, ViZDoom is MIT. Not affiliated with id Software; Doom is a trademark of
id Software LLC.
