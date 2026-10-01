"""U-Net heatmap model + decoder — robot beacon localization project (deep learning part).

Model: input is a 2-channel raster (free-space field, candidate mask); output is, for every raster point,
the logit that "a beacon should be placed here".
Loss: BCE computed only at candidate positions (other positions are meaningless), with positive-class
weighting (the optimal solution is only ~18% of the candidates).

Decoding: the network output is not guaranteed to be feasible, so "guided greedy + pruning + fine-grid
verification" turns it into a valid layout:
  1. In descending order of network probability, add candidates that reduce under-coverage until the coarse grid is satisfied everywhere
  2. Pruning: try removing candidates in ascending order of probability; remove any whose removal keeps the layout feasible
  3. Fine-grid verification; on violations, add the violating points to the active set and go back to 1
The same decoder with "select by gain" is plain greedy, so the two can be compared fairly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_dataset import to_raster_index
from solvers import requirements
from visibility import visibility_matrix


# ------------------------------------------------------------------ model
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net with 3 downsampling levels; the bottleneck's receptive field covers a whole 16x12 map (visibility is
    non-local). Fully convolutional, so it also runs on larger maps."""

    def __init__(self, in_ch=2, base=16):
        super().__init__()
        b = base
        self.enc1, self.enc2, self.enc3, self.enc4 = (
            ConvBlock(in_ch, b), ConvBlock(b, 2 * b), ConvBlock(2 * b, 4 * b), ConvBlock(4 * b, 8 * b))
        self.up3, self.dec3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), ConvBlock(8 * b, 4 * b)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), ConvBlock(4 * b, 2 * b)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), ConvBlock(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):                       # x: (B, 2, H, W)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % 8, 0, -h % 8))    # pad to a multiple of 8 (32x40 for the 16x12 maps)
        e1 = self.enc1(x)                       # full resolution
        e2 = self.enc2(F.max_pool2d(e1, 2))     # 1/2
        e3 = self.enc3(F.max_pool2d(e2, 2))     # 1/4
        e4 = self.enc4(F.max_pool2d(e3, 2))     # 1/8
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)[:, 0, :h, :w]      # (B, H, W) logits


def masked_bce(logits, Y, cand_mask, pos_weight):
    """BCE computed only at candidate positions"""
    loss = F.binary_cross_entropy_with_logits(
        logits, Y, pos_weight=torch.tensor(pos_weight), reduction='none')
    return (loss * cand_mask).sum() / cand_mask.sum().clamp(min=1)


@torch.no_grad()
def predict(model, X):
    """X: (2, H, W) numpy → probability raster (H, W) numpy"""
    model.eval()
    logits = model(torch.from_numpy(X[None]).float())
    return torch.sigmoid(logits)[0].numpy()


def flip_aug(X, Y):
    """Random horizontal/vertical flips (map and label flipped together); geometrically fully valid"""
    if np.random.rand() < 0.5:
        X, Y = X[:, :, ::-1], Y[:, ::-1]
    if np.random.rand() < 0.5:
        X, Y = X[:, ::-1, :], Y[::-1, :]
    return np.ascontiguousarray(X), np.ascontiguousarray(Y)


# ------------------------------------------------------------------ decoding
def cand_probs(prob_raster, cands):
    return np.array([prob_raster[to_raster_index(c)] for c in cands])


def guided_greedy(cands, cells, W1, W2, P_solve, P_check, scores=None,
                  k=3, max_iter=25, mode='score'):
    """Guided greedy + pruning + fine-grid verification.

    scores: static score of each candidate (e.g. network probability). None means select by gain (plain greedy).
    mode:   'score'  — among candidates with gain > 0, pick the highest score (purely network-guided)
            'hybrid' — pick the largest gain × score (network prior × problem structure)
    Returns (list of selected indices, number of iterations)
    """
    M = len(cands)
    A = P_solve.copy()
    V = visibility_matrix(cands, A, cells, W1, W2).astype(bool)
    chosen = []

    def feasible(sel, Vm):
        return bool((Vm[:, sel].sum(1) >= requirements(Vm, k)).all()) if sel \
            else bool((requirements(Vm, k) == 0).all())

    for it in range(1, max_iter + 1):
        # ---- 1. add candidates until the active set is satisfied ----
        deficit = requirements(V, k) - (V[:, chosen].sum(1) if chosen else 0)
        deficit = np.maximum(deficit, 0)
        used = np.zeros(M, bool); used[chosen] = True
        while deficit.max() > 0:
            need = deficit > 0
            gain = (V & need[:, None]).sum(0)
            gain[used] = 0
            if gain.max() == 0:
                break
            if scores is None:
                j = int(gain.argmax())
            elif mode == 'hybrid':
                j = int((gain * scores).argmax())
            else:
                s = np.where(gain > 0, scores, -np.inf)
                j = int(s.argmax())
            used[j] = True; chosen.append(j)
            deficit[V[:, j]] = np.maximum(0, deficit[V[:, j]] - 1)
        # ---- 2. pruning: try removing low-score candidates first ----
        order = (sorted(chosen, key=lambda j: scores[j]) if scores is not None
                 else sorted(chosen, key=lambda j: V[:, j].sum()))
        for j in order:
            trial = [c for c in chosen if c != j]
            if feasible(trial, V):
                chosen = trial
        # ---- 3. fine-grid verification ----
        Vs = visibility_matrix([cands[j] for j in chosen], P_check, cells, W1, W2)
        bad = np.where(Vs.sum(1) < k)[0]
        if len(bad) == 0:
            return chosen, it
        P_bad = P_check[bad]
        V_bad = visibility_matrix(cands, P_bad, cells, W1, W2).astype(bool)
        truly = Vs[bad].sum(1) < np.minimum(k, V_bad.sum(1))   # a violation only if physically satisfiable
        if not truly.any():
            return chosen, it
        V = np.vstack([V, V_bad[truly]])
        A = np.vstack([A, P_bad[truly]])
    return chosen, max_iter
