# Minimum Beacon Placement for Robot Localization

A robot moves freely inside a map built from grid cells. Beacons can be mounted anywhere on the walls, and the robot can
localize itself only when it sees **at least 3 beacons** at the same time; walls block the line of sight. This project
finds the **smallest set of beacons** that lets the robot localize everywhere in the free space, and studies what drives
that number.

## Pipeline

| Step | Notebook | Module | What it does |
|---|---|---|---|
| 1 | `random_map.ipynb` | `map_gen.py` | Random connected maps (growth + hole carving), wall segments, beacon candidates (wall corners + segment midpoints) |
| 2 | `visibility.ipynb` | `visibility.py` | Exact line-of-sight test for beacons mounted on walls; visibility matrix of robot positions × beacons |
| 3 | `optimization.ipynb` | `solvers.py` | Minimum 3-fold set cover: greedy vs. exact ILP (OR-Tools CP-SAT) |
| 4 | `refinement.ipynb` | `solvers.py` | Constraint generation: solve on a coarse grid of robot positions, verify on a fine grid, add violated points until none remain |
| 5 | `analysis.ipynb` | — | Cost of requiring 3 beacons, where optimal beacons go, trimming the candidate set, and a 472-map study of what drives the beacon count |
| 6 | `deep_learning.ipynb` | `dl_dataset.py`, `dl_model.py`, `dl_train.py`, `dl_eval.py` | A U-Net predicts a beacon heatmap from the map; the heatmap is decoded into valid layouts and compared with greedy and the ILP optimum |

## Key results

- **Greedy is clearly suboptimal**: on 10 maps it never matched the ILP optimum, using 4.5 more beacons on average (about 20% more).
- **Solving only on a coarse grid of robot positions is unsafe**: on 5 of 6 maps it underestimated the true minimum by
  1–3 beacons, leaving blind spots. Constraint generation fixes this at a small cost (about 3× faster than modeling the
  fine grid directly on the main map).
- **Requiring 3 visible beacons costs about 3× as many beacons as requiring 1** (3.14× on average): sharing beacons gives no discount.
- **Complexity, not size, drives the cost.** On 472 verified maps, diagonal pinch points (where two free cells touch only
  at a corner) are the best beacon positions (selected 59% of the time), followed by reflex corners (39%). The beacon
  count correlates with the number of reflex corners (r = 0.58) but not with the map area (r = 0.03).
- **The network learns the solver's preferences** (AUC 0.87, same ranking of position types). Hybrid decoding
  (greedy gain × network probability) comes within 1.7 beacons of the optimum on average and reaches it on 19% of test
  maps — significantly better than greedy (sign test p ≈ 2×10⁻⁶). At this map size, though, the exact pipeline
  (about 1.3 s per map) remains the practical choice.

## Running

Install the packages in `requirements.txt`, then run the notebooks in the order above; each one reuses the modules in
this folder.

The data and model files are included, so the notebooks run without regenerating anything:

| File | Contents | Regenerate with |
|---|---|---|
| `visibility_data.npz` | Visibility matrix of the main map (used by step 3) | `visibility.ipynb` |
| `dl_data.npz` | 472 maps (seeds 0–471) with verified optimal layouts | `python dl_dataset.py --n 472` |
| `dl_model.pt`, `dl_model_hist.npz` | Trained U-Net and its training curves | `python dl_train.py --epochs 30 --base 8` |
| `dl_eval.npz` | Test-set evaluation | `python dl_eval.py` |

`dl_dataset.py` and `dl_eval.py` resume from an existing output file, so delete it first to start from scratch.

## Limitations

- Robot positions are checked on a finite grid of sample points; a rigorous guarantee for every point of the free space
  would need a visibility decomposition.
- Beacon candidates are limited to wall corners and segment midpoints.
- CP-SAT runs with 8 workers, so re-running can return a different (equally optimal) layout; beacon counts are unaffected.
- The deep-learning dataset is small (376 training maps), and the model starts to overfit after about 13 epochs.
