"""Train the heatmap model — robot beacon localization project (deep learning part).

Usage: python dl_train.py --data dl_data.npz --epochs 60 --out dl_model.pt
Split by seed: seed % 10 == 0 → test set, == 1 → validation set, everything else → training set.
"""

import argparse
import time

import numpy as np
import torch

from dl_model import UNet, flip_aug, masked_bce


def split(seeds):
    s = np.asarray(seeds) % 10
    return np.where((s != 0) & (s != 1))[0], np.where(s == 1)[0], np.where(s == 0)[0]


def precision_at_n(prob, Y, C):
    """For each map, take the n highest-probability candidates (n = size of the optimal solution) and count how many are in the optimal solution"""
    hits, total = 0, 0
    for p, y, c in zip(prob, Y, C):
        n = int(y.sum())
        if n == 0:
            continue
        pc = np.where(c > 0, p, -1).ravel()
        top = np.argsort(-pc)[:n]
        hits += int(y.ravel()[top].sum()); total += n
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dl_data.npz')
    ap.add_argument('--out', default='dl_model.pt')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base', type=int, default=8)
    ap.add_argument('--wd', type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d = np.load(args.data)
    X, Y = d['X'].astype(np.float32), d['Y'].astype(np.float32)
    tr, va, te = split(d['seed'])
    C = X[:, 1]                                          # candidate mask
    n_pos, n_cand = Y[tr].sum(), C[tr].sum()
    pos_weight = float((n_cand - n_pos) / n_pos)
    print(f'train {len(tr)} / val {len(va)} / test {len(te)}   '
          f'pos_weight={pos_weight:.2f}  (optimal solution = {100 * n_pos / n_cand:.1f}% of candidates)')

    model = UNet(base=args.base)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'UNet parameters: {n_param:,}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    Xv, Yv, Cv = map(torch.from_numpy, (X[va], Y[va], C[va]))
    hist = {'train_loss': [], 'val_loss': [], 'val_p_at_n': []}
    best, t0 = 1e9, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = zip(*[flip_aug(X[j], Y[j]) for j in idx])
            xb, yb = torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))
            loss = masked_bce(model(xb), yb, xb[:, 1], pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            lv = model(Xv)
            vloss = masked_bce(lv, Yv, Cv, pos_weight).item()
            pv = torch.sigmoid(lv).numpy()
        p_at_n = precision_at_n(pv, Y[va], C[va])
        hist['train_loss'].append(tot / len(tr)); hist['val_loss'].append(vloss)
        hist['val_p_at_n'].append(p_at_n)
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), args.out)
        if ep % 5 == 0 or ep == 1:
            print(f'ep {ep:3d}  train {tot / len(tr):.4f}  val {vloss:.4f}  '
                  f'val P@n {p_at_n:.3f}  ({time.time() - t0:.0f}s)', flush=True)
    np.savez(args.out.replace('.pt', '_hist.npz'), **{k: np.array(v) for k, v in hist.items()},
             pos_weight=pos_weight)
    print(f'best val loss {best:.4f}  → {args.out}')


if __name__ == '__main__':
    main()
