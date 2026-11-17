import os

import numpy as np

from common import OUT
from game import (ATTACK, BACK, FWD, LEFT, MONSTERS, RIGHT, RUN, TICS, inside, new_episode, new_game, open_wad,
                  player_pos, wad_lumps, wad_map, wall_dist, yaw_change)

R, STRIDE = 8, 4  # raster step in map units; nodes every R * STRIDE = 32
STEP_UP, HEAD, RADIUS = 24, 56, 18  # highest step, player height, player radius 16 + 2
TELEPORTS = {39, 97, 125, 126}
CHAINSAW = 2005  # kept out of reach: picked up, it stays the weapon and melee fights last long
SOLID_RADIUS = {2035: 10, 2028: 16, 85: 16, 86: 16, 30: 16, 31: 16, 32: 16, 33: 16, 35: 16, 36: 16, 37: 16, 41: 16,
                42: 16, 43: 16, 44: 16, 45: 16, 46: 16, 47: 16, 48: 16, 54: 32, 55: 16, 56: 16, 57: 16, 70: 16,
                25: 16, 26: 16, 27: 16, 28: 16, 29: 16, 49: 16, 50: 16, 51: 16, 52: 16, 53: 16, 72: 16}
WALK, LOOK, PAUSE, FIGHT, RECOVER = range(5)
TOUR_CELL = 448
SPAWN_MIN_NODES = 200


def grid_path(level):
    return os.path.join(OUT, f"nav_{level.lower()}.npz")


def build_grid(level="E1M1"):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    wad = open_wad()
    w, lump = wad_map(level, wad), wad_lumps(level, wad)
    v = np.frombuffer(lump["VERTEXES"], np.int16).reshape(-1, 2).astype(np.float64)
    ld = np.frombuffer(lump["LINEDEFS"], np.uint16).reshape(-1, 7)
    sec_tag = np.frombuffer(lump["SECTORS"], np.uint8).reshape(-1, 26)[:, 24:26].copy().view(np.int16)[:, 0]
    fl, ce = w["floor"], w["ceil"]
    nb = [set() for _ in fl]
    for a, b in w["links"]:
        nb[a].add(b)
        nb[b].add(a)
    block = np.array([(*v[v1], *v[v2]) for v1, v2, flags, *_, left in ld if left == 0xFFFF or flags & 1])
    g = new_game(0, level=level, wad=wad)
    new_episode(g)
    start = player_pos(g)
    g.close()
    home = [i for i, ls in enumerate(w["lines"]) if len(ls) >= 3 and inside(start[:1], start[1:], ls)[0]]
    conn, todo = set(home), list(home)  # sectors joined to the start; the rest are hidden control sectors
    while todo:
        for t in nb[todo.pop()] - conn:
            conn.add(t)
            todo.append(t)
    allv = np.array([ln for ls in w["lines"] for ln in ls])
    x0, y0 = allv[:, [0, 2]].min(), allv[:, [1, 3]].min()
    nx, ny = int((allv[:, [0, 2]].max() - x0) // R) + 1, int((allv[:, [1, 3]].max() - y0) // R) + 1
    gx, gy = np.meshgrid(x0 + R * (np.arange(nx) + .5), y0 + R * (np.arange(ny) + .5))
    px, py = gx.ravel(), gy.ravel()
    sec = np.full(len(px), -1, np.int32)
    for s in sorted(conn):
        ls = np.array(w["lines"][s])
        if len(ls) < 3:
            continue
        sel = np.flatnonzero((px >= ls[:, [0, 2]].min()) & (px <= ls[:, [0, 2]].max())
                             & (py >= ls[:, [1, 3]].min()) & (py <= ls[:, [1, 3]].max()))
        sec[sel[inside(px[sel], py[sel], w["lines"][s])]] = s
    dist = np.full(len(px), -1.0)
    ins = np.flatnonzero(sec >= 0)
    for i in range(0, len(ins), 4096):
        dist[ins[i:i + 4096]] = wall_dist(px[ins[i:i + 4096]], py[ins[i:i + 4096]], block)
    th = np.frombuffer(lump["THINGS"], np.int16).reshape(-1, 5)
    for x, y, _, typ, flags in th:
        if (typ in SOLID_RADIUS or typ == CHAINSAW) and flags & 2:
            near = np.flatnonzero((np.abs(px - x) < 80) & (np.abs(py - y) < 80))
            r = SOLID_RADIUS.get(typ, 48)
            dist[near] = np.minimum(dist[near], np.maximum(np.hypot(px[near] - x, py[near] - y) - r, 0))
    floor = np.where(sec >= 0, fl[sec], np.nan)
    walk = (sec >= 0) & (np.where(sec >= 0, ce[sec], 0) - np.where(sec >= 0, fl[sec], 0) >= HEAD) & (dist >= RADIUS)
    sec, dist, floor, walk = (a.reshape(ny, nx) for a in (sec, dist, floor, walk))
    iy, ix = np.meshgrid(np.arange(STRIDE // 2, ny, STRIDE), np.arange(STRIDE // 2, nx, STRIDE), indexing="ij")
    ok = walk[iy, ix]
    node = np.full(iy.shape, -1, np.int64)
    node[ok] = np.arange(ok.sum())
    nyx = np.stack([iy[ok], ix[ok]], 1)
    rows, cols, cost = [], [], []
    t = np.linspace(0, 1, 2 * STRIDE + 1)
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        a = node[max(0, -dy):iy.shape[0] - max(0, dy), max(0, -dx):iy.shape[1] - max(0, dx)]
        b = node[max(0, dy):iy.shape[0] - max(0, -dy) or None, max(0, dx):iy.shape[1] - max(0, -dx) or None]
        m = (a >= 0) & (b >= 0)
        a, b = a[m], b[m]
        sy = np.rint(nyx[a, 0][:, None] + t[None] * (nyx[b, 0] - nyx[a, 0])[:, None]).astype(int)
        sx = np.rint(nyx[a, 1][:, None] + t[None] * (nyx[b, 1] - nyx[a, 1])[:, None]).astype(int)
        good = walk[sy, sx].all(1) & (np.diff(floor[sy, sx], axis=1) <= STEP_UP).all(1)
        d = dist[sy, sx].min(1)
        rows.append(a[good])
        cols.append(b[good])
        cost.append(R * STRIDE * np.hypot(dy, dx) * (1 + 1.5 * np.clip((48 - d[good]) / 48, 0, 1)))
    xy = np.stack([x0 + R * (nyx[:, 1] + .5), y0 + R * (nyx[:, 0] + .5)], 1)
    tree, dest = cKDTree(xy), th[th[:, 3] == 14]
    for v1, v2, flags, special, tag, r, left in ld:
        if special in TELEPORTS and tag:
            for x, y, *_ in dest:
                if any(inside(np.array([x], float), np.array([y], float), w["lines"][s])[0]
                       for s in np.flatnonzero(sec_tag == tag)):
                    src = tree.query_ball_point((v[v1] + v[v2]) / 2, 48)
                    rows.append(np.array(src, np.int64))
                    cols.append(np.full(len(src), tree.query([x, y])[1], np.int64))
                    cost.append(np.full(len(src), 64.0))
    rows, cols, cost = np.concatenate(rows), np.concatenate(cols), np.concatenate(cost)
    G = csr_matrix((cost, (rows, cols)), shape=(len(xy), len(xy)))
    _, lab = connected_components(G, directed=True, connection="strong")
    tours = make_tours(G, xy, lab)
    np.savez_compressed(grid_path(level), x0=x0, y0=y0, dist=dist, floor=floor, walk=walk, xy=xy,
                        rows=rows, cols=cols, cost=cost,
                        tour=np.concatenate(tours), tour_len=np.array([len(t) for t in tours]))
    print(f"{level}: {len(xy)} walkable nodes, {len(rows)} edges, tours of {[len(t) for t in tours]} waypoints",
          flush=True)


def make_tours(G, xy, lab, min_part=100):
    from scipy.sparse.csgraph import dijkstra
    tours = []
    for p in np.flatnonzero(np.bincount(lab) >= min_part):
        nodes = np.flatnonzero(lab == p)
        key = (xy[nodes] // TOUR_CELL).astype(int)
        wps = []
        for k in np.unique(key, axis=0):
            m = nodes[(key == k).all(1)]
            wps.append(m[np.argmin(np.hypot(*(xy[m] - xy[m].mean(0)).T))])
        wps = np.array(wps)
        D = dijkstra(G, indices=wps)[:, wps]
        D = (D + D.T) / 2
        order, left = [0], set(range(1, len(wps)))
        while left:
            j = min(left, key=lambda q: D[order[-1], q])
            order.append(j)
            left.remove(j)
        order, better = np.array(order), True
        while better:
            better = False
            for i in range(1, len(order) - 1):
                a, b = order[i - 1], order[i]
                c, d = order[i:], np.r_[order[i + 1:], order[0]]
                gain = D[a, b] + D[c, d] - D[a, c] - D[b, d]
                j = int(np.argmax(gain))
                if gain[j] > 1e-6:
                    order[i:i + j + 1] = order[i:i + j + 1][::-1]
                    better = True
        tours.append(wps[order])
    return tours


class Grid:
    def __init__(self, level="E1M1"):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree
        if not os.path.exists(grid_path(level)):
            build_grid(level)
        d = np.load(grid_path(level))
        self.x0, self.y0 = float(d["x0"]), float(d["y0"])
        self.walk, self.floor, self.dist = d["walk"], d["floor"], d["dist"]
        self.xy = d["xy"]
        self.G = csr_matrix((d["cost"], (d["rows"], d["cols"])), shape=(len(self.xy), len(self.xy)))
        self.tree = cKDTree(self.xy)
        self.part = connected_components(self.G, directed=True, connection="strong")[1]
        self.tours = {int(self.part[t[0]]): t for t in np.split(d["tour"], np.cumsum(d["tour_len"])[:-1])}

    def clear(self, p, q):
        n = int(np.hypot(*(q - p)) // R) + 2
        t = np.linspace(0, 1, n)[:, None]
        s = p[None] + t * (q - p)[None]
        iy = np.clip(((s[:, 1] - self.y0) // R).astype(int), 0, self.walk.shape[0] - 1)
        ix = np.clip(((s[:, 0] - self.x0) // R).astype(int), 0, self.walk.shape[1] - 1)
        return bool(self.walk[iy, ix][1:].all() and (np.diff(self.floor[iy, ix]) <= STEP_UP).all())

    def free(self, p, angles, far=256):
        th = np.radians(np.asarray(angles, np.float64))[:, None]
        d = np.arange(R, far + R, R)[None]
        iy = np.clip(((p[1] + np.sin(th) * d - self.y0) // R).astype(int), 0, self.walk.shape[0] - 1)
        ix = np.clip(((p[0] + np.cos(th) * d - self.x0) // R).astype(int), 0, self.walk.shape[1] - 1)
        blocked = ~self.walk[iy, ix]
        return np.where(blocked.any(1), blocked.argmax(1), d.shape[1]) * R

    def node_at(self, p):
        _, idx = self.tree.query(p, k=8)
        for i in idx:
            if self.clear(np.asarray(p, np.float64), self.xy[i]):
                return int(i)
        return int(idx[0])


class Navigator:
    def __init__(self, rng, level="E1M1"):
        self.rng, self.g = rng, Grid(level)
        self.visits = np.zeros(len(self.g.xy))
        self.unreachable = np.zeros(len(self.g.xy), bool)
        self.blocked_until = np.zeros(len(self.g.xy))
        self.t = 0
        self.reset()

    def spawn_cells(self, cells):
        return cells[np.bincount(self.g.part)[self.g.part[self.g.tree.query(cells)[1]]] >= SPAWN_MIN_NODES]

    def reset(self):
        self.hist, self.moving, self.pressed, self.queue, self.path, self.respawn = [], [], [], [], None, False
        self.tour, self.next_wp, self.direction = None, 0, 1
        self.reached = 0
        self.enemy_gone, self.engage, self.strafe_left = 99, True, 0
        self.fight_len, self.target, self.turn = 0, None, 0.0

    def _style(self):
        r = self.rng
        self.run = r.random() < .55
        self.max_turn = r.uniform(4, 9)
        self.strafey = r.random() < .3
        self.curve = r.uniform(.3, .8)

    def _plan(self, pos):
        from scipy.sparse import diags
        from scipy.sparse.csgraph import dijkstra
        r, src = self.rng, self.g.node_at(pos)
        G, off = self.g.G, self.blocked_until > self.t
        off[src] = False
        if off.any():
            G = diags((~off).astype(float)) @ G @ diags((~off).astype(float))
            G.eliminate_zeros()
        dist, pred = dijkstra(G, indices=src, return_predecessors=True)
        reach = np.isfinite(dist) & ~self.unreachable
        part = int(self.g.part[src])
        if self.tour is None or self.g.part[self.tour[0]] != part:
            self.tour = self.g.tours.get(part)
            if self.tour is not None:
                self.next_wp = int(np.argmin(np.hypot(*(self.g.xy[self.tour] - pos).T)))
                self.direction = int(r.choice([-1, 1]))
        goal = None
        if self.tour is not None:
            for _ in range(len(self.tour)):
                k = int(self.tour[self.next_wp])
                if reach[k] and np.hypot(*(self.g.xy[k] - pos)) > 96:
                    goal = k
                    break
                self.next_wp = (self.next_wp + self.direction) % len(self.tour)
        if goal is None:
            cand = np.flatnonzero(reach & (dist > 256) & (dist < 2500))
            if not len(cand):
                cand = np.flatnonzero(reach & (dist > 64))
            if not len(cand):
                return False
            p = 1 / (1 + self.visits[cand]) ** 2
            goal = int(r.choice(cand, p=p / p.sum()))
        nodes = [goal]
        while nodes[-1] != src and pred[nodes[-1]] >= 0:
            nodes.append(pred[nodes[-1]])
        self.goal, self.nodes = goal, np.array(nodes[::-1])
        self.path = self.g.xy[self.nodes]
        self.k, self.best_k, self.since = 0, 0, 0
        self._style()
        return True

    def _stand(self, n, mode):
        self.queue += [(np.zeros(7, np.float32), 0.0, mode)] * n

    def _turn_open(self, pos, angle):
        r, dirs = self.rng, np.arange(0, 360, 22.5)
        free = self.g.free(pos, dirs)
        err = yaw_change(angle, float(r.choice(dirs[free >= .8 * free.max()])))
        rate = r.uniform(4, 6) * np.sign(err or 1)
        self.queue += [(np.zeros(7, np.float32), rate, LOOK)] * int(round(abs(err) / (TICS * abs(rate))))

    def _arrive(self, pos, angle):
        r, u = self.rng, self.rng.random()
        if self.tour is not None:
            self.next_wp = (self.next_wp + self.direction) % len(self.tour)
            self.reached += 1
        if u < .01:
            rate = r.uniform(3, 6) * r.choice([-1, 1])
            n = int(r.uniform(90, 360) / (TICS * abs(rate)))
            self.queue += [(np.zeros(7, np.float32), rate, LOOK)] * n
            self._turn_open(pos, angle + n * TICS * rate)
        elif u < .023:
            self._turn_open(pos, angle)
            self._stand(int(r.integers(15, 51)) if u < .02 else int(r.integers(35, 141)), PAUSE)
        self.path = None

    def __call__(self, pos, angle, labels=(), width=640):
        pos = np.asarray(pos, np.float64)
        self.t += 1
        self.hist.append(pos)
        self.visits[self.g.tree.query(pos)[1]] += 1
        lap = self.tour is not None and self.reached >= len(self.tour) or self.path is not None and self.tour is None
        self.respawn = lap or len(self.hist) > 100 and sum(self.pressed[-100:]) > 50 \
            and np.abs(np.array(self.hist[-100:]) - pos).max() < 1
        b, turn, mode = self._act(pos, angle, labels, width)
        self.turn = turn = .5 * self.turn + .5 * turn
        moved = bool(b[[FWD, BACK, LEFT, RIGHT]].any())
        self.moving.append(mode != FIGHT and moved)
        self.pressed.append(moved)
        return b, turn, mode

    def _act(self, pos, angle, labels, width):
        r = self.rng
        b = np.zeros(7, np.float32)
        enemies = [lb for lb in labels if lb.object_name.lower() in MONSTERS and lb.width > .03 * width]
        if enemies and self.enemy_gone >= 10:
            self.engage, self.fight_len = r.random() < .85, 0
        self.enemy_gone = 0 if enemies else self.enemy_gone + 1
        if enemies and self.engage:
            self.fight_len += 1
            self.engage = self.fight_len < 60
            self.queue, self.path = [], None
            same = [lb for lb in enemies if lb.object_id == self.target]
            e = same[0] if same else max(enemies, key=lambda lb: lb.width * lb.height)
            self.target = e.object_id
            off = (e.x + e.width / 2) / width - .5
            err = np.degrees(np.arctan(2 * off))
            if self.strafe_left <= 0:
                self.strafe_left, self.strafe = int(r.integers(15, 41)), r.choice([-1, 0, 0, 0, 1])
            self.strafe_left -= 1
            if self.strafe:
                b[LEFT if self.strafe > 0 else RIGHT] = 1
            b[ATTACK] = float(abs(off) < .07 and r.random() < .6)
            b[FWD] = float(e.width < .06 * width and r.random() < .5)
            b[BACK] = float(e.width > .2 * width and r.random() < .5)
            b[RUN] = float(r.random() < .5)
            return b, float(np.clip(-.3 * err, -8, 8)) if abs(err) > 1 else 0.0, FIGHT
        if self.queue:
            b, turn, mode = self.queue.pop(0)
            return b.copy(), turn, mode
        if self.path is None and not self._plan(pos):
            walk = np.zeros(7, np.float32)
            walk[FWD] = 1
            rate = r.uniform(4, 8) * r.choice([-1, 1])
            self.queue += [(np.zeros(7, np.float32), rate, RECOVER)] * int(r.integers(6, 20)) \
                + [(walk, 0.0, RECOVER)] * int(r.integers(15, 40))
            return self._act(pos, angle, labels, width)
        def dk(j):
            return np.hypot(*(self.path[j] - pos))
        while self.k < len(self.path) - 1 and (dk(self.k) < 56 or dk(self.k + 1) < dk(self.k)):
            self.k += 1
        if self.k >= len(self.path) - 1 and dk(-1) < 56:
            self._arrive(pos, angle)
            return self._act(pos, angle, labels, width)
        if self.k > self.best_k:
            self.best_k, self.since = self.k, 0
        self.since += 1
        blocked = len(self.moving) >= 15 and all(self.moving[-15:]) and np.hypot(*(pos - self.hist[-15])) < 12
        if blocked or self.since > 70:
            self.moving = []
            back = np.zeros(7, np.float32)
            back[BACK] = 1
            self.queue += [(back, r.uniform(2, 5) * r.choice([-1, 1]), RECOVER)] * int(r.integers(6, 13))
            if self.since > 70:
                self.unreachable[self.g.tree.query_ball_point(self.g.xy[self.goal], 64)] = True
            else:
                self.blocked_until[self.nodes[self.k:self.k + 2]] = self.t + 2100
            if self.tour is not None:
                self.next_wp = (self.next_wp + self.direction) % len(self.tour)
                self.reached += 1
            self.path = None
            b, turn, mode = self.queue.pop(0)
            return b.copy(), turn, mode
        tgt = self.path[self.k]
        for p in self.path[self.k + 1:self.k + 7][::-1]:
            if self.g.clear(pos, p):
                tgt = p
                break
        err = yaw_change(angle, np.degrees(np.arctan2(*(tgt - pos)[::-1])))
        turn = float(np.clip(.3 * err, -self.max_turn, self.max_turn))
        near = np.hypot(*(tgt - pos)) < 120
        if abs(err) < 35 or (abs(err) < 80 and not near and r.random() < self.curve):
            b[FWD] = 1
        if self.strafey and 15 < abs(err) < 45 and not near:
            b[LEFT if err > 0 else RIGHT] = 1
        b[RUN] = float(self.run)
        mode = WALK
        b[ATTACK] = float(r.random() < .005)
        if r.random() < .001:
            self._stand(int(r.integers(8, 36)), PAUSE)
        return b, turn, mode
