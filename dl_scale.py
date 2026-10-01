"""Scaling experiment — robot beacon localization project (deep learning part).

The network was trained on 16x12 maps. On maps of growing size, its heatmap is used to prune the candidate set before
the exact solver, and compared with the exact solver itself and with a simple non-learned ranking:
  exact_plain  constraint generation as first implemented (reuse=False)
  exact        constraint generation with cached verification and warm starts — the reference optimum
  nn_<f>       exact solver over the network's top fraction f of the candidates (repaired wherever a point needs more)
  prior_<f>    the same, ranking candidates by type instead (pinch > reflex > midpoint > convex, rates from training maps)
  hybrid       hybrid decoding from deep_learning.ipynb (no ILP)
For each map and method: beacons, wall-clock time (everything after map generation, including the network's forward
pass) and candidates kept. The methods run in a random order per map, so slow-downs during a long run favour none.

Usage: python dl_scale.py --budget 600      # can be rerun repeatedly to resume
"""

import argparse
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from dl_dataset import (GRID_H, GRID_W, RES_CHECK, RES_SOLVE, map_params, rasterize_input,
                        to_raster_index, unique_candidates)
from dl_model import UNet, cand_probs, guided_greedy, predict
from dl_train import split
from map_gen import cand_kind, make_map
from solvers import constraint_generation
from visibility import sample_points, wall_arrays

SIZES = [(16, 12), (24, 18), (32, 24)]
N_MAPS = {(16, 12): 20, (24, 18): 12, (32, 24): 10}
FRACS = (0.2, 0.3, 0.45)


def type_prior(data):
    """ILP selection rate of each candidate type on the training maps"""
    avail, chosen = Counter(), Counter()
    tr, _, _ = split(data['seed'])
    for i in tr:
        seed = int(data['seed'][i])
        cells, segs, corners, mids = make_map(GRID_W, GRID_H, *map_params(seed), seed)
        for p in unique_candidates(corners, mids):
            kind = cand_kind(p, cells)
            avail[kind] += 1
            chosen[kind] += int(data['Y'][i][to_raster_index(p)] > 0.5)
    return {kind: chosen[kind] / avail[kind] for kind in avail}


def auc(score, label):
    """AUC = probability that a random positive scores higher than a random negative (rank statistic)"""
    pos, neg = score[label], score[~label]
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def run_map(model, prior, w, h, seed):
    target, holes = map_params(seed, (w * h) / (GRID_W * GRID_H))
    cells, segs, corners, mids = make_map(w, h, target, holes, seed)
    cands = unique_candidates(corners, mids)
    W1, W2 = wall_arrays(segs)
    P_s, P_c = sample_points(cells, RES_SOLVE), sample_points(cells, RES_CHECK)
    M = len(cands)
    tie = np.random.default_rng(seed).uniform(0, 1e-6, M)       # random tie-break within a candidate type

    def nn_scores():
        return cand_probs(predict(model, rasterize_input(cells, cands, w, h)), cands)

    def prior_scores():
        return np.array([prior.get(cand_kind(p, cells), 0.0) for p in cands]) + tie

    def exact(reuse):
        return constraint_generation(cands, cells, W1, W2, P_s, P_c, reuse=reuse)

    def pruned(scores_fn, f):
        s = scores_fn()
        keep = np.argsort(-s, kind='stable')[:round(f * M)]
        return constraint_generation(cands, cells, W1, W2, P_s, P_c, keep=keep, scores=s)

    def hybrid():
        sol, _ = guided_greedy(cands, cells, W1, W2, P_s, P_c, scores=nn_scores(), mode='hybrid')
        return sol, None

    methods = {'exact_plain': lambda: exact(False), 'exact': lambda: exact(True), 'hybrid': hybrid}
    for f in FRACS:
        methods[f'nn_{f}'] = lambda f=f: pruned(nn_scores, f)
        methods[f'prior_{f}'] = lambda f=f: pruned(prior_scores, f)
    names = list(methods)
    random.Random(seed).shuffle(names)

    r = {'w': w, 'h': h, 'seed': seed, 'area': len(cells), 'n_cand': M, 'n_walls': len(segs), 'n_points': len(P_s)}
    for name in names:
        t0 = time.time()
        sol, log = methods[name]()
        entry = {'n': len(sol), 't': time.time() - t0, 'sel': np.array(sol), 'kept': M}
        if log is not None:                       # constraint generation: where the time went
            entry.update(kept=log[-1]['kept'], iters=len(log),
                         **{part: sum(e[f't_{part}'] for e in log) for part in ('init', 'ilp', 'verify', 'update')})
        r[name] = entry

    # how well each ranking finds the exact solution's beacons (not timed)
    label = np.zeros(M, dtype=bool)
    label[r['exact']['sel']] = True
    for name, s in [('nn', nn_scores()), ('prior', prior_scores())]:
        r[f'auc_{name}'] = auc(s, label)
        top = np.argsort(-s, kind='stable')
        r[f'recall_{name}'] = {f: label[top[:round(f * M)]].sum() / label.sum() for f in FRACS}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dl_data.npz')
    ap.add_argument('--model', default='dl_model.pt')
    ap.add_argument('--out', default='dl_scale.npz')
    ap.add_argument('--base', type=int, default=8)
    ap.add_argument('--budget', type=float, default=None, help='seconds; save and exit once exceeded')
    args = ap.parse_args()

    model = UNet(base=args.base)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    prior = type_prior(np.load(args.data))
    print('type prior (training maps):', {k: round(v, 3) for k, v in sorted(prior.items(), key=lambda kv: -kv[1])})

    results = list(np.load(args.out, allow_pickle=True)['results']) if os.path.exists(args.out) else []
    done = {(r['w'], r['h'], r['seed']) for r in results}
    todo = [(w, h, 10 * i) for (w, h) in SIZES for i in range(N_MAPS[(w, h)]) if (w, h, 10 * i) not in done]
    print(f'{len(done)} maps done, {len(todo)} to go')

    t0 = time.time()
    for w, h, seed in todo:
        r = run_map(model, prior, w, h, seed)
        results.append(r)
        np.savez(args.out, results=np.array(results, dtype=object))
        ex = r['exact']
        print(f"  {w}x{h} seed {seed:3d}: exact {ex['n']:3d} in {ex['t']:5.1f}s (plain {r['exact_plain']['t']:5.1f}s) | "
              + ' | '.join(f"{m} +{r[m]['n'] - ex['n']} {r[m]['t']:4.1f}s" for m in ('nn_0.3', 'nn_0.45', 'hybrid')),
              flush=True)
        if args.budget and time.time() - t0 > args.budget:
            print('Time budget used up; saved. Run again to resume.')
            break


if __name__ == '__main__':
    main()
