"""Training dataset generation — robot beacon localization project (deep learning part).

Each sample = one random map + the optimal beacon layout verified by constraint generation.
Both the map and the labels are rasterized at 2x resolution: raster (i, j) corresponds to world
coordinate (i/2, j/2), so wall corners (integer coordinates) and wall-segment midpoints
(half-integer coordinates) land exactly on raster points.

Input channels:
  ch0 free-space field — fraction of free cells among the cells adjacent to the raster point
                         (cell center 1/0, on a wall face 0.5, reflex corner 0.75, convex corner 0.25)
  ch1 candidate mask   — whether the point is a beacon candidate position
Label:
  Y   optimal-solution mask — whether the candidate is in the optimal layout

Usage:
  python dl_dataset.py --n 600 --out dl_data.npz            # generate / resume up to 600 samples
  python dl_dataset.py --n 600 --out dl_data.npz --budget 40 # run for only 40 s, then save and exit (can be rerun repeatedly)
"""

import argparse
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from map_gen import make_map
from solvers import constraint_generation
from visibility import sample_points, wall_arrays

GRID_W, GRID_H = 16, 12
RES_SOLVE, RES_CHECK = 4, 8
RH, RW = 2 * GRID_H + 1, 2 * GRID_W + 1        # raster size (rows = y, cols = x)


# ------------------------------------------------------------------ geometry → raster
def map_params(seed, scale=1.0):
    """Derive the map's area and number of holes from the seed, for diversity and reproducibility.
    scale multiplies both, for maps on a larger grid (scale = grid area / 16x12 area)."""
    rng = random.Random(seed * 7919 + 13)
    return round(rng.randint(60, 130) * scale), round(rng.randint(4, 14) * scale)


def unique_candidates(corners, mids):
    """Deduplicated candidate list (a T-junction may be both a corner and a midpoint; visibility depends only on position)"""
    seen, out = set(), []
    for p in list(corners) + list(mids):
        key = (float(p[0]), float(p[1]))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def to_raster_index(p):
    """World coordinate (x, y) → raster (row, col) = (2y, 2x)"""
    return int(round(2 * p[1])), int(round(2 * p[0]))


def rasterize_free(cells, W=GRID_W, H=GRID_H):
    """Free-space field, shape (2H+1, 2W+1)"""
    F = np.zeros((H + 2, W + 2), dtype=np.float32)      # one ring of zero padding
    for (x, y) in cells:
        F[y + 1, x + 1] = 1.0
    R = np.zeros((2 * H + 1, 2 * W + 1), dtype=np.float32)
    for r in range(2 * H + 1):
        ys = [r // 2] if r % 2 == 1 else [r // 2 - 1, r // 2]
        for c in range(2 * W + 1):
            xs = [c // 2] if c % 2 == 1 else [c // 2 - 1, c // 2]
            R[r, c] = np.mean([F[y + 1, x + 1] for y in ys for x in xs])
    return R


def rasterize_input(cells, cands, W=GRID_W, H=GRID_H):
    """Network input for one map: (free-space field, candidate mask), shape (2, 2H+1, 2W+1)"""
    X = np.zeros((2, 2 * H + 1, 2 * W + 1), dtype=np.float32)
    X[0] = rasterize_free(cells, W, H)
    for p in cands:
        X[1][to_raster_index(p)] = 1.0
    return X


def build_sample(seed):
    """Generate one sample. Returns a dict, or None on failure"""
    try:
        return _build_sample(seed)
    except Exception as e:                       # one failed map must not take down the whole batch
        print(f'  seed {seed} failed: {type(e).__name__}: {e}', flush=True)
        return None


def _build_sample(seed):
    target, holes = map_params(seed)
    cells, segs, corners, mids = make_map(GRID_W, GRID_H, target, holes, seed)
    cands = unique_candidates(corners, mids)
    W1, W2 = wall_arrays(segs)
    P_solve = sample_points(cells, RES_SOLVE)
    P_check = sample_points(cells, RES_CHECK)
    sol, log = constraint_generation(cands, cells, W1, W2, P_solve, P_check)

    X = rasterize_input(cells, cands)
    Y = np.zeros((RH, RW), dtype=np.float32)
    for j in sol:
        Y[to_raster_index(cands[j])] = 1.0
    return {'seed': seed, 'X': X, 'Y': Y, 'n_beacons': len(sol),
            'area': len(cells), 'n_cand': len(cands), 'iters': len(log)}


# ------------------------------------------------------------------ main
def load(path):
    if not os.path.exists(path):
        return {'seed': [], 'X': [], 'Y': [], 'n_beacons': [], 'area': [],
                'n_cand': [], 'iters': []}
    d = np.load(path)
    return {k: list(d[k]) for k in d.files}


def save(path, data):
    np.savez_compressed(path, **{k: np.array(v) for k, v in data.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--out', default='dl_data.npz')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--budget', type=float, default=None, help='seconds; save and exit once exceeded')
    args = ap.parse_args()

    data = load(args.out)
    done = set(int(s) for s in data['seed'])
    todo = [s for s in range(args.n) if s not in done]
    print(f'{len(done)} samples already exist, {len(todo)} still needed')
    if not todo:
        return

    t0 = time.time()
    chunk = args.workers * 2
    with Pool(args.workers) as pool:
        for i in range(0, len(todo), chunk):
            res = pool.map(build_sample, todo[i:i + chunk])
            for r in res:
                if r is None:
                    continue
                for k in data:
                    data[k].append(r[k])
            save(args.out, data)
            el = time.time() - t0
            print(f'  {len(data["seed"]):4d} samples  ({el:5.1f}s, '
                  f'{len(data["seed"]) - len(done)} new)', flush=True)
            if args.budget and el > args.budget:
                print('Time budget used up; saved. Run again to resume.')
                break
    print('Done' if len(data['seed']) >= args.n else 'Not finished (can be resumed)')


if __name__ == '__main__':
    main()
