# Math 547 Final Project — SINDy/DMD Control of Human Motion Capture

Identifies sparse governing equations and minimum-actuator control for human mocap data using the framework $\dot{x} = f(x) + Bu$, where $f$ is identified via SINDy in a DMD-reduced latent space and $B$ is the minimum input matrix for controllability.

---

## Repo structure

```
Math547_FinalProject/
│
├── data/
│   ├── train/          # 15 training trials (.npy + .mat)
│   │   ├── jumping_1 … jumping_5
│   │   ├── running_1  … running_5
│   │   └── walking_1  … walking_5
│   └── test/           # 3 held-out trials (one per class)
│       ├── jumping_1t
│       ├── running_1t
│       └── walking_1t
│
├── project_utils/
│   ├── dataset_utils.py   # data loading helpers
│   ├── MTDMD.py           # Multi-Trajectory DMD implementation
│   └── __init__.py
│
├── PCAc.ipynb             # POD/PCA analysis and latent projections
├── DMDc.ipynb             # DMD analysis, energy plots, mode animations
├── SINDy.ipynb            # SINDy identification + controllability  ← main new notebook
│
├── sindy_writeup.tex      # LaTeX writeup: methods, results, figure captions
├── sindy_writeup.pdf      # Compiled PDF (11 pages)
│
├── figs/                  # All saved figures (PNG)
└── anim/                  # Saved animations (GIF)
```

---

## Data

Each file is a `.npy` array of shape **(114, 100)**:
- **114 features** = 38 skeletal joints × 3 Cartesian axes
- **100 timesteps** per trial

Three action classes, 5 training trials each (15 total), 1 test trial each (3 total).

`load_mat(folder, name)` returns shape `(114, 100)`.  
`get_dataset_tensor(folder, names)` returns shape `(N, 114, 100)` — stack of all trials.  
For MTDMD/SINDy, transpose to `(N, 100, 114)` so rows are time snapshots.

---

## `project_utils/`

### `dataset_utils.py`

| Function | Returns | Shape |
|---|---|---|
| `load_vec(folder, name)` | flattened 1D array | `(114×100,)` |
| `load_mat(folder, name)` | 2D matrix | `(114, 100)` |
| `get_dataset_mat(folder, names)` | stacked flat vecs | `(N, 114×100)` |
| `get_dataset_tensor(folder, names)` | stacked matrices | `(N, 114, 100)` |
| `get_train_test_set(...)` | `(train, test)` as mat or tensor | — |

### `MTDMD.py`

Implements **Multi-Trajectory DMD** (Anzaki et al., 2024).

`mtdmd(experiment_timeseries, variance_threshold=0.95)`

- **Input:** `(N, T, n)` array — N trials × T timesteps × n features
- **Output dict:**

| Key | Shape | Description |
|---|---|---|
| `"A"` | `(r, r)` | Reduced discrete-time dynamics matrix $A_{\text{dmd}}$ |
| `"basis"` | `(n, r)` | Real orthogonal SVD basis $U_r$ (latent coordinate system) |
| `"eigenvalues"` | `(r,)` | DMD eigenvalues |
| `"modes"` | `(n, r)` | DMD spatial modes $U_r W$ (complex) |
| `"rank"` | int | Chosen rank $r$ |
| `"singular_values"` | `(n,)` | All singular values of the Hessian $H$ |
| `"reconstructions"` | `(N, T, n)` | Per-trial DMD reconstructions |
| `"mean_reconstruction"` | `(T, n)` | Mean reconstruction across trials |
| `"mode_amplitudes"` | list of `(r, T)` | Per-trial temporal mode coefficients |

Rank $r$ is chosen as the smallest integer such that the top-$r$ singular values of $H = \sum_\mu X_\mu X_\mu^\top$ account for at least `variance_threshold` of total energy.

---

## Notebooks

### `PCAc.ipynb`
- Loads all 15 training trials as a flat matrix
- Computes PCA (POD) modes and cumulative energy
- Projects data onto top 2 and 3 POD modes (2D/3D scatter plots)
- Finds class centroids in $k$-dimensional POD space
- **Outputs:** `figs/PCA_Energy.png`, `figs/PCA_proj2.png`, `figs/PCA_proj3.png`, `figs/PCA_corr.png`, `figs/PCA_centroids_proj.png`, `figs/PCA_samples_proj.png`

### `DMDc.ipynb`
- Runs MTDMD on all 15 training trials
- Plots cumulative energy and singular value spectrum
- Animates DMD modes and per-class mean reconstructions
- Computes covariance/correlation matrix in DMD latent space
- **Outputs:** `figs/DMD_Energy.png`, `anim/DMD_Mode_*.gif`, `anim/Low_Dim_*.gif`, `anim/Jumping_*.gif`, `anim/Running_*.gif`, `anim/Walking_*.gif`

### `SINDy.ipynb` ← **main deliverable**
Runs the full $\dot{x} = f(x) + Bu$ pipeline end-to-end.

| Cell | What it does |
|---|---|
| Data load | Loads all 15 trials; prints shape |
| MTDMD | Runs at 97% variance → rank $r=3$, basis $U_r \in \mathbb{R}^{114\times3}$, $A_{\text{dmd}} \in \mathbb{R}^{3\times3}$ |
| Energy plot | Cumulative Hessian energy curve |
| Latent projection | Computes $z_\mu(t) = U_r^\top x_\mu(t)$ for all 15 trials |
| Latent traj plot | 3D plot of all 15 trajectories coloured by class |
| SINDy prep | Stacks 1,485 (state, increment) pairs; standardises by $\sigma_k$ |
| SINDy fit | Degree-2 polynomial library, STLSQ threshold=0.005, fit to all 15 at once |
| SINDy score | $R^2 = 13.6\%$ — reports how much of $\dot{z}$ is autonomous vs. control-driven |
| Derivative scatter | True vs. predicted $\dot{\tilde{z}}_k$ scatter, one panel per mode |
| Extract $A$ | Pulls linear block from SINDy $\Xi$; compares to $A_{\text{dmd}} - I$ |
| Eigenvalue plot | $A_{\text{dmd}}$ eigenvalues in complex plane with unit circle |
| Controllability | Greedy search for min $m$ s.t. $\text{rank}\,\mathcal{C}(A_{\text{dmd}},B) = r$ |
| Ctrl rank plot | Rank vs. number of inputs |
| Lift $B$ | $B_{\text{sensor}} = U_r B_{\text{latent}} \in \mathbb{R}^{114}$ |
| $B$ plot | Bar chart of actuator weights over 114 features |

**Outputs:** `figs/SINDy_energy.png`, `figs/SINDy_latent_traj.png`, `figs/SINDy_deriv_scatter.png`, `figs/SINDy_eigenvalues.png`, `figs/SINDy_ctrl_rank.png`, `figs/SINDy_B_sensor.png`

---

## Key results

| Quantity | Value |
|---|---|
| State dimension $n$ | 114 |
| Latent rank $r$ (at 97% variance) | **3** |
| Energy captured at $r=3$ | 99.98% |
| $A_{\text{dmd}}$ eigenvalues | $0.9994,\ 0.9997 \pm 0.0049j$ |
| SINDy library | Degree-2 polynomial, $p=10$ terms |
| SINDy $R^2$ (derivative prediction) | 13.6% |
| Minimum inputs $m^*$ | **1** |
| Actuated latent mode | 0 (any of {0,1,2} works) |

The low $R^2$ is physically meaningful: ~86% of $\dot{z}$ is driven by active muscle control ($Bu$), not autonomous flow ($f$). A single scalar control signal suffices to steer the full 3D latent state between action classes.

---

## Python environment

Notebooks use the `sindy_py_547` virtual environment:

```bash
# activate
source sindy_py_547/bin/activate

# key packages
pysindy==2.1.0
numpy, matplotlib, scipy, scikit-learn, ipykernel
```

The kernel is registered as `sindy_py_547` in Jupyter.  
`PCAc.ipynb` and `DMDc.ipynb` use the system Python environment (no special venv required beyond standard scientific stack).

---

## Writeup

`sindy_writeup.tex` / `sindy_writeup.pdf` — 11-page LaTeX document covering:
- Full mathematical derivations of MTDMD, SINDy, and minimum-input controllability
- All numerical results with exact values
- A description of every figure and what it shows
- Analysis: why $R^2 = 14\%$ is the right answer, why $m^* = 1$, comparison of $A_{\text{sindy}}$ vs $A_{\text{dmd}}$

Compile with:
```bash
pdflatex sindy_writeup.tex && pdflatex sindy_writeup.tex
```
