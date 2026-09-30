"""Evaluate the heatmap model on the test set — robot beacon localization project (deep learning part).

For each test map:
  - network prediction → NN-guided greedy decoding (+ pruning + fine-grid verification) → beacon count
  - hybrid (gain × network probability) and plain greedy (by gain) through the same decoder → beacon count
  - optimal (ILP verified by constraint generation) → beacon count (the dataset label); the exact
    pipeline is re-run once per map to time it against the decoders
Also records (predicted probability, type, whether in the optimal solution) for every candidate,
to analyze what the network has learned.

Usage: python dl_eval.py --budget 40      # can be rerun repeatedly to resume
"""

import argparse
import os
import time

import numpy as np
import torch

from dl_dataset import (GRID_H, GRID_W, RES_CHECK, RES_SOLVE, map_params,
                        to_raster_index, unique_candidates)
from dl_model import UNet, cand_probs, guided_greedy, predict
from dl_train import split
from map_gen import cand_kind, make_map
from solvers import constraint_generation
from visibility import sample_points, wall_arrays

KINDS = ['pinch', 'reflex', 'midpoint', 'convex', 'other']


def eval_one(model, seed, X, Y):
    target, holes = map_params(seed)
    cells, segs, corners, mids = make_map(GRID_W, GRID_H, target, holes, seed)
    cands = unique_candidates(corners, mids)
    W1, W2 = wall_arrays(segs)
    P_s, P_c = sample_points(cells, RES_SOLVE), sample_points(cells, RES_CHECK)

    prob = predict(model, X)
    pc = cand_probs(prob, cands)
    t0 = time.time(); nn_sol, nn_it = guided_greedy(cands, cells, W1, W2, P_s, P_c, scores=pc); t_nn = time.time() - t0
    t0 = time.time(); gr_sol, gr_it = guided_greedy(cands, cells, W1, W2, P_s, P_c, scores=None); t_gr = time.time() - t0
    t0 = time.time(); hy_sol, hy_it = guided_greedy(cands, cells, W1, W2, P_s, P_c, scores=pc, mode='hybrid'); t_hy = time.time() - t0
    t0 = time.time(); ex_sol, _ = constraint_generation(cands, cells, W1, W2, P_s, P_c); t_opt = time.time() - t0
    opt = int(Y.sum())

    kinds = np.array([KINDS.index(cand_kind(c, cells)) for c in cands])
    in_opt = np.array([Y[to_raster_index(c)] > 0.5 for c in cands])
    return {
        'seed': seed, 'opt': opt, 'nn': len(nn_sol), 'greedy': len(gr_sol), 'hybrid': len(hy_sol),
        'nn_iters': nn_it, 'greedy_iters': gr_it, 'hybrid_iters': hy_it,
        't_nn': t_nn, 't_greedy': t_gr, 't_hybrid': t_hy, 't_opt': t_opt, 'opt_resolved': len(ex_sol),
        'area': len(cells), 'n_cand': len(cands),
        'cand_prob': pc, 'cand_kind': kinds, 'cand_in_opt': in_opt,
        'nn_sel': np.array(nn_sol), 'greedy_sel': np.array(gr_sol), 'hybrid_sel': np.array(hy_sol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dl_data.npz')
    ap.add_argument('--model', default='dl_model.pt')
    ap.add_argument('--out', default='dl_eval.npz')
    ap.add_argument('--base', type=int, default=8)
    ap.add_argument('--budget', type=float, default=None)
    args = ap.parse_args()

    d = np.load(args.data)
    _, _, te = split(d['seed'])
    model = UNet(base=args.base)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    results = []
    if os.path.exists(args.out):
        results = list(np.load(args.out, allow_pickle=True)['results'])
    done = {int(r['seed']) for r in results}
    todo = [i for i in te if int(d['seed'][i]) not in done]
    print(f'test set: {len(te)} maps, {len(done)} evaluated, {len(todo)} remaining')

    t0 = time.time()
    for i in todo:
        r = eval_one(model, int(d['seed'][i]), d['X'][i].astype(np.float32), d['Y'][i])
        results.append(r)
        print(f"  seed {r['seed']:4d}: opt={r['opt']:2d}  nn={r['nn']:2d}  hybrid={r['hybrid']:2d}  greedy={r['greedy']:2d}",
              flush=True)
        np.savez(args.out, results=np.array(results, dtype=object))
        if args.budget and time.time() - t0 > args.budget:
            print('Time budget used up; saved.'); break

    if len(results) == len(te):
        opt = np.array([r['opt'] for r in results]); nn = np.array([r['nn'] for r in results])
        gr = np.array([r['greedy'] for r in results]); hy = np.array([r['hybrid'] for r in results])
        print(f'\n{len(results)} test maps:  mean opt {opt.mean():.2f} | nn {nn.mean():.2f} '
              f'(+{(nn - opt).mean():.2f}) | hybrid {hy.mean():.2f} (+{(hy - opt).mean():.2f}) '
              f'| greedy {gr.mean():.2f} (+{(gr - opt).mean():.2f})')
        print(f'share reaching the optimum: nn {100 * (nn == opt).mean():.0f}%,  hybrid {100 * (hy == opt).mean():.0f}%,  '
              f'greedy {100 * (gr == opt).mean():.0f}%')
        t = {k: np.mean([r[k] for r in results]) for k in ('t_opt', 't_nn', 't_hybrid', 't_greedy')}
        print(f'mean time per map: exact {t["t_opt"]:.2f}s | nn {t["t_nn"]:.2f}s | '
              f'hybrid {t["t_hybrid"]:.2f}s | greedy {t["t_greedy"]:.2f}s')
        print(f'exact re-solve matches the label on '
              f'{sum(int(r["opt_resolved"]) == int(r["opt"]) for r in results)}/{len(results)} maps')


if __name__ == '__main__':
    main()
