import os
import struct
import tempfile

import numpy as np
import vizdoom as vzd
from PIL import Image

from common import OUT

TICS = 2
BUTTONS = [vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
           vzd.Button.SPEED, vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.TURN_LEFT_RIGHT_DELTA]
FWD, BACK, LEFT, RIGHT, RUN, ATTACK, USE = range(7)
MONSTERS = {m.lower() for m in ("ZombieMan", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul",
                                "Cacodemon", "BaronOfHell", "HellKnight", "Revenant", "Fatso", "Arachnotron",
                                "PainElemental", "Archvile")}
WAD = os.path.join(os.path.dirname(vzd.__file__), "freedoom1.wad")
OPEN_WAD = os.path.join(OUT, "freedoom1_open.wad")


def new_game(seed=0, depth=False, labels=False, infinite_ammo=False, level="E1M1", wad=WAD):
    g = vzd.DoomGame()
    g.set_doom_config_path(os.path.join(tempfile.gettempdir(), "_vizdoom.ini"))
    g.set_doom_game_path(wad)
    g.set_doom_map(level)
    g.set_window_visible(False)
    g.set_audio_buffer_enabled(False)
    g.set_sound_enabled(False)
    g.set_render_hud(False)
    g.set_render_crosshair(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_depth_buffer_enabled(depth)
    g.set_labels_buffer_enabled(labels)
    g.set_available_buttons(BUTTONS)
    g.set_available_game_variables([vzd.GameVariable.ANGLE])
    # no centre prints such as "A secret is revealed!" on the frames
    flags = "+sv_cheats 1 +cl_showsecretmessage 0 +con_midtime 0"
    g.add_game_args(flags + (" +sv_infiniteammo 1" if infinite_ammo else ""))
    g.set_doom_skill(3)
    g.set_seed(seed)
    g.init()
    return g


def new_episode(g):
    g.new_episode()
    g.send_game_command("god")
    g.send_game_command("give keys")  # else "BLUE KEY NEEDED FOR THIS DOOR." is printed on the frames


def downscale(screen, hw):
    return np.asarray(Image.fromarray(screen).resize((hw[1], hw[0]), Image.BOX))


def yaw_change(a0, a1):
    return (a1 - a0 + 180.0) % 360.0 - 180.0


def inside(px, py, lines):
    L = np.array(lines, np.float64)
    x1, y1, x2, y2 = (L[:, i][None] for i in range(4))
    X, Y = px[:, None], py[:, None]
    xint = x1 + (Y - y1) * (x2 - x1) / np.where(y2 == y1, 1e-9, y2 - y1)
    return ((((y1 > Y) != (y2 > Y)) & (X < xint)).sum(1) % 2) == 1


def lump_dir(data, name):
    _, n, off = struct.unpack_from("<4sii", data, 0)
    ents = [struct.unpack_from("<ii8s", data, off + 16 * i) for i in range(n)]
    names = [e[2].rstrip(b"\0").decode() for e in ents]
    i0 = names.index(name)
    return {names[j]: ents[j][:2] for j in range(i0 + 1, i0 + 11)}


def wad_lumps(name="E1M1", wad=WAD):
    with open(wad, "rb") as f:
        data = f.read()
    return {k: data[o:o + n] for k, (o, n) in lump_dir(data, name).items()}


MANUAL_DOORS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
DOOR_LINES = MANUAL_DOORS | {2, 4, 29, 46, 61, 63, 86, 90, 99, 103, 105, 106, 108, 109, 111, 112, 114, 115, 133, 134,
                             135, 136, 137} | {3, 16, 42, 50, 75, 76, 107, 110, 113, 116}


def open_wad(levels=("E1M1", "E1M2", "E1M3")):
    if os.path.exists(OPEN_WAD):
        return OPEN_WAD
    data = bytearray(open(WAD, "rb").read())
    for lv in levels:
        d = lump_dir(data, lv)
        (lo, ln), (so, sn), (eo, en) = d["LINEDEFS"], d["SIDEDEFS"], d["SECTORS"]
        ld = np.frombuffer(bytes(data[lo:lo + ln]), np.uint16).reshape(-1, 7)
        sides = np.frombuffer(bytes(data[so:so + sn]), np.uint8).reshape(-1, 30)
        side_sec = sides[:, 28:30].copy().view(np.int16)[:, 0]
        sec = np.frombuffer(bytes(data[eo:eo + en]), np.uint8).reshape(-1, 26)
        ceil, tag = (sec[:, k:k + 2].copy().view(np.int16)[:, 0] for k in (2, 24))
        nb = [set() for _ in sec]
        for *_, r, left in ld:
            if left != 0xFFFF and side_sec[r] != side_sec[left]:
                nb[side_sec[r]].add(int(side_sec[left]))
                nb[side_sec[left]].add(int(side_sec[r]))
        doors = set()
        for i, (v1, v2, flags, special, t, r, left) in enumerate(ld):
            if special in DOOR_LINES:
                if special in MANUAL_DOORS and left != 0xFFFF:
                    doors.add(int(side_sec[left]))
                elif t:
                    doors |= {int(s) for s in np.flatnonzero(tag == t)}
                data[lo + 14 * i + 6:lo + 14 * i + 8] = b"\0\0"
        for s in doors:
            if nb[s]:
                c = min(int(ceil[t]) for t in nb[s]) - 4
                if c > ceil[s]:
                    data[eo + 26 * s + 2:eo + 26 * s + 4] = int(c).to_bytes(2, "little", signed=True)
    os.makedirs(os.path.dirname(OPEN_WAD), exist_ok=True)
    with open(OPEN_WAD + ".tmp", "wb") as f:
        f.write(data)
    os.replace(OPEN_WAD + ".tmp", OPEN_WAD)
    return OPEN_WAD


def wad_map(name="E1M1", wad=WAD):
    lump = wad_lumps(name, wad)
    v = np.frombuffer(lump["VERTEXES"], np.int16).reshape(-1, 2).astype(np.float64)
    ld = np.frombuffer(lump["LINEDEFS"], np.uint16).reshape(-1, 7)
    side_sec = np.frombuffer(lump["SIDEDEFS"], np.uint8).reshape(-1, 30)[:, 28:30].copy().view(np.int16)[:, 0]
    sec = np.frombuffer(lump["SECTORS"], np.uint8).reshape(-1, 26)
    floor, ceil = (sec[:, k:k + 2].copy().view(np.int16)[:, 0].astype(np.float64) for k in (0, 2))
    lines, links, walls = [[] for _ in range(len(sec))], [], []
    for v1, v2, _, _, _, r, left in ld:
        seg = (v[v1, 0], v[v1, 1], v[v2, 0], v[v2, 1])
        sr, sl = side_sec[r], (side_sec[left] if left != 0xFFFF else -1)
        if sr != sl:
            lines[sr].append(seg)
            if sl >= 0:
                lines[sl].append(seg)
        if sl >= 0:
            links.append((sr, sl))
        else:
            walls.append(seg)
    return dict(lines=lines, floor=floor, ceil=ceil, links=links, walls=np.array(walls))


def wall_dist(px, py, walls):
    x1, y1, x2, y2 = (walls[:, i][None] for i in range(4))
    dx, dy = x2 - x1, y2 - y1
    u = np.clip(((px[:, None] - x1) * dx + (py[:, None] - y1) * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px[:, None] - x1 - u * dx, py[:, None] - y1 - u * dy).min(1)


def player_pos(g):
    return np.array([g.get_game_variable(vzd.GameVariable.POSITION_X),
                     g.get_game_variable(vzd.GameVariable.POSITION_Y)])


_FLOOR = {}


def on_floor(px, py, min_gap=56, margin=20, level="E1M1"):
    if level not in _FLOOR:
        w = wad_map(level)
        g = new_game(0, level=level)
        new_episode(g)
        start = (g.get_game_variable(vzd.GameVariable.POSITION_X), g.get_game_variable(vzd.GameVariable.POSITION_Y))
        g.close()
        parent = list(range(len(w["lines"])))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in w["links"]:
            parent[root(a)] = root(b)
        sx, sy = np.array([start[0]]), np.array([start[1]])
        home = [i for i, ls in enumerate(w["lines"]) if ls and inside(sx, sy, ls)[0]]
        assert len(home) == 1, home
        ok = [i for i, ls in enumerate(w["lines"]) if len(ls) >= 3 and root(i) == root(home[0])
              and w["ceil"][i] - w["floor"][i] >= min_gap]
        _FLOOR[level] = (w, ok, start)
    w, ok, _ = _FLOOR[level]
    px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
    hit = np.zeros(len(px), bool)
    for i in ok:
        hit |= inside(px, py, w["lines"][i])
    return hit & (wall_dist(px, py, w["walls"]) >= margin)


def map_cells(cell=64, level="E1M1"):
    on_floor(np.zeros(1), np.zeros(1), level=level)
    w, _, start = _FLOOR[level]
    allv = np.array([ln for ls in w["lines"] for ln in ls])
    x0, y0, x1, y1 = allv[:, [0, 2]].min(), allv[:, [1, 3]].min(), allv[:, [0, 2]].max(), allv[:, [1, 3]].max()
    gx, gy = np.meshgrid(np.arange(x0, x1, cell) + cell / 2, np.arange(y0, y1, cell) + cell / 2)
    px, py = gx.ravel(), gy.ravel()
    ok = on_floor(px, py, level=level)
    return np.stack([px[ok], py[ok]], 1), w["lines"], start


def spawn(g, rng, cells, cell, level="E1M1"):
    for _ in range(20):
        x, y = cells[rng.integers(len(cells))] + rng.uniform(-cell / 3, cell / 3, 2)
        if not on_floor([x], [y], level=level)[0]:
            continue
        g.send_game_command(f"warp {x:.0f} {y:.0f}")
        g.make_action([0.0] * 8, 1)  # the warp happens on the next tic
        p0, free = player_pos(g), False
        for k in (FWD, BACK, LEFT, RIGHT):
            b = [0.0] * 8
            b[k] = 1.0
            g.make_action(b, 2)
            if np.hypot(*(player_pos(g) - p0)) > 4:
                free = True
                break
        if free:
            break
    g.make_action([0.0] * 7 + [float(rng.uniform(-180, 180))], 1)


def record(n, seed, level, seg=1000, reset_every=2, hw=(96, 128), cell=64):
    from bot import Navigator
    cells = map_cells(cell, level)[0]
    g = new_game(seed, depth=True, labels=True, infinite_ammo=True, level=level, wad=open_wad())
    rng = np.random.default_rng(seed)
    player = Navigator(rng, level)
    cells = player.spawn_cells(cells)
    out = dict(frames=np.zeros((n, *hw, 3), np.uint8), actions=np.zeros((n, 8), np.float32),
               episodes=np.zeros(n, np.int32), masks=np.zeros((n, *hw), bool), pos=np.zeros((n, 2), np.float32),
               angle=np.zeros(n, np.float32), mode=np.zeros(n, np.int8))
    noop = [0.0] * 8
    ep, left = -1, 0
    for i in range(n):
        if left == 0:
            ep += 1
            if ep % reset_every == 0 or g.is_episode_finished():
                new_episode(g)
                g.make_action(noop, 4)  # the first tics of an episode ignore input
            spawn(g, rng, cells, cell, level)
            player.reset()
            left = seg
        s = g.get_state()
        a0 = s.game_variables[0]
        out["frames"][i], out["episodes"][i] = downscale(s.screen_buffer, hw), ep
        out["pos"][i], out["angle"][i] = player_pos(g), a0
        m = np.isin(s.labels_buffer, [lb.value for lb in s.labels if lb.object_name.lower() in MONSTERS])
        out["masks"][i] = downscale(m.astype(np.uint8) * 255, hw) >= 128
        b, turn, out["mode"][i] = player(out["pos"][i], a0, s.labels)
        left = 1 if player.respawn else left
        g.make_action(b.tolist() + [-turn], TICS)
        left -= 1
        if g.is_episode_finished():
            out["actions"][i], left = np.concatenate([b, [0.0]]), 0
            continue
        out["actions"][i] = np.concatenate([b, [yaw_change(a0, g.get_state().game_variables[0])]])
    g.close()
    return out
