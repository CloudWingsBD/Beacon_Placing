"""Visibility computation — robot beacon localization project.

Reusable module extracted from visibility.ipynb: line-of-sight wall-crossing test + visibility matrix.

A robot at p sees a beacon b on a wall if and only if:
  (A) b faces the side where p is (side probe); and
  (B) segment pb intersects no wall except at its endpoint b.
Grazing a wall corner / running collinear along a wall is always conservatively treated as blocked.
"""

import numpy as np

__all__ = ["wall_arrays", "sample_points", "visible_from",
           "visibility_matrix", "EPS_SIDE"]

EPS_SIDE = 1e-6   # side-probe step (only used for the cell-membership lookup)


def wall_arrays(segments):
    """Split wall segments into endpoint arrays (W1, W2), each of shape (K, 2)."""
    return (np.array([s[0] for s in segments], dtype=float),
            np.array([s[1] for s in segments], dtype=float))


def sample_points(cells, res=4):
    """Take res x res sample points inside each free cell (strictly in the cell interior)."""
    offs = [(i + 0.5) / res for i in range(res)]
    pts = [(x + dx, y + dy) for (x, y) in sorted(cells) for dx in offs for dy in offs]
    return np.array(pts, dtype=float)


def _cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _on_seg_bbox(ax, ay, bx, by, qx, qy):
    return ((np.minimum(ax, bx) <= qx) & (qx <= np.maximum(ax, bx)) &
            (np.minimum(ay, by) <= qy) & (qy <= np.maximum(ay, by)))


def visible_from(b, P, cells, W1, W2):
    """Visibility of beacon b from the sample point set P; returns a boolean array (N,)."""
    bx, by = float(b[0]), float(b[1])
    Px, Py = P[:, 0], P[:, 1]

    # (A) side check: b nudged toward p must land in free space
    dx, dy = Px - bx, Py - by
    dist = np.hypot(dx, dy)
    ok = dist > 0
    qx = bx + EPS_SIDE * np.divide(dx, dist, out=np.zeros_like(dx), where=ok)
    qy = by + EPS_SIDE * np.divide(dy, dist, out=np.zeros_like(dy), where=ok)
    cx, cy = np.floor(qx).astype(int), np.floor(qy).astype(int)
    side_ok = ok & np.array([(int(a), int(c)) in cells for a, c in zip(cx, cy)])

    # (B) the line of sight does not cross any wall
    w1x, w1y = W1[:, 0][None, :], W1[:, 1][None, :]
    w2x, w2y = W2[:, 0][None, :], W2[:, 1][None, :]
    px, py = Px[:, None], Py[:, None]

    d1 = _cross(w1x, w1y, w2x, w2y, px, py)                        # (N, K)
    d2 = _cross(W1[:, 0], W1[:, 1], W2[:, 0], W2[:, 1], bx, by)    # (K,)
    d3 = _cross(px, py, bx, by, w1x, w1y)                          # (N, K)
    d4 = _cross(px, py, bx, by, w2x, w2y)                          # (N, K)

    proper = (((d1 > 0) & (d2[None, :] < 0)) | ((d1 < 0) & (d2[None, :] > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))
    t1 = (d1 == 0) & _on_seg_bbox(w1x, w1y, w2x, w2y, px, py)
    t2 = (d2 == 0)[None, :] & _on_seg_bbox(W1[:, 0], W1[:, 1],
                                           W2[:, 0], W2[:, 1], bx, by)[None, :]
    t3 = (d3 == 0) & _on_seg_bbox(px, py, bx, by, w1x, w1y)
    t4 = (d4 == 0) & _on_seg_bbox(px, py, bx, by, w2x, w2y)
    blocked_generic = proper | t1 | t2 | t3 | t4

    contains_b = ((d2 == 0) &
                  _on_seg_bbox(W1[:, 0], W1[:, 1], W2[:, 0], W2[:, 1], bx, by))
    blocked = np.where(contains_b[None, :], d1 == 0, blocked_generic)

    return side_ok & ~blocked.any(axis=1)


def visibility_matrix(beacons, P, cells, W1, W2):
    """Visibility matrix V[i, j] = whether sample point i can see beacon j, shape (N, M)."""
    return np.column_stack([visible_from(b, P, cells, W1, W2) for b in beacons])
