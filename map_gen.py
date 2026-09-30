"""Random map generation — robot beacon localization project.

Reusable module extracted from random_map.ipynb.

Map representation: the free space is a set of cells; cell (x, y) occupies the square [x,x+1]x[y,y+1].
Every cell edge between a free cell and a non-free cell (including outside the map) is a wall.
"""

import random
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

__all__ = [
    "neighbors", "grow_map", "is_connected", "carve_holes",
    "extract_wall_edges", "merge_edges", "beacon_candidates", "cand_kind",
    "make_map", "plot_map",
]


# ---------------------------------------------------------------- generation
def neighbors(c):
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def grow_map(w, h, target, rng):
    """Growth method: expand randomly from a seed cell; returns the set of free cells (4-connected by construction)."""
    target = min(target, w * h)
    start = (w // 2, h // 2)
    cells = {start}
    frontier = {n for n in neighbors(start) if 0 <= n[0] < w and 0 <= n[1] < h}
    while len(cells) < target and frontier:
        c = rng.choice(sorted(frontier))
        frontier.discard(c)
        cells.add(c)
        for n in neighbors(c):
            if 0 <= n[0] < w and 0 <= n[1] < h and n not in cells:
                frontier.add(n)
    return cells


def is_connected(cells):
    """BFS check that the free space is 4-connected."""
    if not cells:
        return False
    start = next(iter(cells))
    seen, dq = {start}, deque([start])
    while dq:
        for n in neighbors(dq.popleft()):
            if n in cells and n not in seen:
                seen.add(n)
                dq.append(n)
    return len(seen) == len(cells)


def carve_holes(cells, n_holes, rng):
    """Remove up to n_holes interior cells to form holes, checking connectivity at every step."""
    cells, removed = set(cells), set()
    attempts = 0
    while len(removed) < n_holes and attempts < n_holes * 30:
        attempts += 1
        interior = [c for c in cells
                    if all(nb in cells or nb in removed for nb in neighbors(c))]
        if not interior:
            break
        adj = [c for c in interior if any(nb in removed for nb in neighbors(c))]
        pool = adj if adj and rng.random() < 0.7 else interior
        c = rng.choice(sorted(pool))
        if len(cells) > 1 and is_connected(cells - {c}):
            cells.discard(c)
            removed.add(c)
    return cells


# ---------------------------------------------------------------- walls
def extract_wall_edges(cells):
    """Return all unit wall edges ((x1,y1),(x2,y2)); endpoints are integer lattice points."""
    edges = []
    for (x, y) in cells:
        if (x, y - 1) not in cells: edges.append(((x, y), (x + 1, y)))
        if (x, y + 1) not in cells: edges.append(((x, y + 1), (x + 1, y + 1)))
        if (x - 1, y) not in cells: edges.append(((x, y), (x, y + 1)))
        if (x + 1, y) not in cells: edges.append(((x + 1, y), (x + 1, y + 1)))
    return edges


def _merge_intervals(ivs):
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a == out[-1][1]:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def merge_edges(edges):
    """Merge collinear, connected unit walls into long wall segments."""
    horiz, vert = {}, {}
    for p1, p2 in edges:
        if p1[1] == p2[1]:
            horiz.setdefault(p1[1], []).append((min(p1[0], p2[0]), max(p1[0], p2[0])))
        else:
            vert.setdefault(p1[0], []).append((min(p1[1], p2[1]), max(p1[1], p2[1])))
    segments = []
    for y, ivs in horiz.items():
        segments += [((a, y), (b, y)) for a, b in _merge_intervals(ivs)]
    for x, ivs in vert.items():
        segments += [((x, a), (x, b)) for a, b in _merge_intervals(ivs)]
    return segments


def beacon_candidates(segments):
    """Beacon candidates: wall corners (segment endpoints) + the midpoint of every segment. Returns (corners, mids)."""
    corners, mids = set(), []
    for p1, p2 in segments:
        corners.add(p1)
        corners.add(p2)
        mids.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
    return sorted(corners), mids


def cand_kind(p, cells):
    """Candidate type: 'reflex' / 'convex' corner, wall 'midpoint', or 'pinch' point.

    A pinch point is a lattice point where two free cells touch only diagonally. It lies inside two
    crossing wall segments (so it is a candidate only as a segment midpoint), and a beacon there is
    visible from both free cells.
    """
    x, y = p
    if abs(x - round(x)) > 1e-9 or abs(y - round(y)) > 1e-9:
        return 'midpoint'
    px, py = int(round(x)), int(round(y))
    q = [(px - 1, py - 1) in cells, (px, py - 1) in cells, (px - 1, py) in cells, (px, py) in cells]
    f = sum(q)
    if f == 2:
        return 'pinch' if q[0] == q[3] else 'midpoint'
    return {1: 'convex', 3: 'reflex'}.get(f, 'other')


def make_map(w=16, h=12, target=120, holes=10, seed=0):
    """Full pipeline: generate -> carve holes -> wall segments -> candidates.

    Returns (cells, segments, corners, mids).
    """
    rng = random.Random(seed)
    cells = grow_map(w, h, target, rng)
    cells = carve_holes(cells, holes, rng)
    assert is_connected(cells), "free space must be connected"
    segments = merge_edges(extract_wall_edges(cells))
    corners, mids = beacon_candidates(segments)
    return cells, segments, corners, mids


# ---------------------------------------------------------------- visualization
def plot_map(cells, segments=None, corners=None, mids=None, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for (x, y) in cells:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#dbeafe",
                               edgecolor="#eff6ff", linewidth=0.5))
    if segments:
        for (x1, y1), (x2, y2) in segments:
            ax.plot([x1, x2], [y1, y2], color="#1f2937",
                    linewidth=2.5, solid_capstyle="round")
    if corners:
        xs, ys = zip(*corners)
        ax.scatter(xs, ys, s=32, color="#dc2626", zorder=3, label="corner candidates")
    if mids:
        xs, ys = zip(*mids)
        ax.scatter(xs, ys, s=20, color="#f59e0b", marker="s",
                   zorder=3, label="midpoint candidates")
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    ax.set_xlim(min(xs) - 1, max(xs) + 2)
    ax.set_ylim(min(ys) - 1, max(ys) + 2)
    ax.set_aspect("equal")
    ax.set_title(title)
    if corners or mids:
        ax.legend(loc="upper right", fontsize=8)
    return ax
