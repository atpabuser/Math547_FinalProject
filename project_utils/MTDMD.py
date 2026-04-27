"""
Multi-Trajectory Dynamic Mode Decomposition (MTDMD)
Based on: Anzaki et al., "Multi-trajectory Dynamic Mode Decomposition" (2024)

Input
-----
experiment_timeseries : np.ndarray, shape (n_expr, n_ts, n_vars)
    Sensor measurements — n_expr experiments × n_ts timesteps × n_vars sensors.

Outputs
-------
- DMD eigenvalues and spatial modes
- Per-experiment reconstructed trajectories (n_ts × n_vars)
- Mean reconstructed trajectory (n_ts × n_vars)
- DMD mode temporal evolution coefficients
- Chosen rank r (data-driven, 95% variance threshold)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_trajectory_pair(data):
    """
    Given data of shape (T, d), build the DMD snapshot matrices:
      X  = data[0 : T-1, :].T   shape (d, T-1)   — "present" snapshots
      X' = data[1 : T,   :].T   shape (d, T-1)   — "future"  snapshots
    """
    X  = data[:-1, :].T   # (d, m)
    Xp = data[1:,  :].T   # (d, m)
    return X, Xp


# ---------------------------------------------------------------------------
# Core MTDMD
# ---------------------------------------------------------------------------

def mtdmd(experiment_timeseries, variance_threshold = 0.95):
    """
    Fit MTDMD (no control) to multiple experimental trajectories.

    Implements Eq. (45) with B=0:

        A* = (Σ_µ  Y_µ X_µᵀ) · (Σ_ν  X_ν X_νᵀ)⁺

    Dimension reduction follows Appendix B (Eq. B1–B2): the rank r is chosen
    so that the top-r singular values of the summed Hessian (Σ X_µ X_µᵀ)
    explain at least `variance_threshold` of total variance.

    Parameters
    ----------
    experiment_timeseries : np.ndarray, shape (N_expr, n_ts, n_vars)
        N_expr  — number of independent experiments
        n_ts  — number of timesteps
        n_vars  — number of sensors / spatial degrees of freedom
    variance_threshold : float
        Fraction of variance (in the Hessian singular values) that the
        retained modes must explain. Default 0.95.

    Returns
    -------
    dict with keys
        "A"              : np.ndarray (r, r)                — reduced DMD operator
        "eigenvalues"    : np.ndarray (r,)                  — DMD eigenvalues
        "modes"          : np.ndarray (n_vars, r)           — spatial DMD modes (columns)
        "rank"           : int                              — chosen rank r
        "singular_values": np.ndarray (n_vars,)             — all singular values of Hessian
        "reconstructions": np.ndarray (N_expr,n_ts,n_vars)  — per-experiment reconstructions
        "mean_reconstruction" : np.ndarray (n_ts,n_vars)    — mean reconstruction over experiments
        "mode_amplitudes": list[np.ndarray]                 — per-experiment temporal coefficients, shape (r, n_ts)
    """
    N_expr, n_ts, n_vars = experiment_timeseries.shape

    # ------------------------------------------------------------------
    # Step 1: Build snapshot pairs for every trajectory
    # ------------------------------------------------------------------
    Xs, Xps = [], []
    for mu in range(N_expr):
        X_mu, Xp_mu = _build_trajectory_pair(experiment_timeseries[mu])
        Xs.append(X_mu)    # (n_vars, m)  where m = n_ts-1
        Xps.append(Xp_mu)  # (n_vars, m)

    # ------------------------------------------------------------------
    # Step 2: Accumulate Hessian H = Σ_ν  X_ν X_νᵀ  (n_vars × n_vars)
    #         and Jacobian  J = Σ_µ  Y_µ X_µᵀ       (n_vars × n_vars)
    #
    # These correspond to Eq. (46)–(47) with B=0, so:
    #   H = Σ X_µ X_µᵀ
    #   J = Σ Xp_µ X_µᵀ   (Y_µ = Xp_µ in the no-control case)
    # ------------------------------------------------------------------
    H = np.zeros((n_vars, n_vars))   # Hessian
    J = np.zeros((n_vars, n_vars))   # (negative half) Jacobian

    for X_mu, Xp_mu in zip(Xs, Xps):
        H  += X_mu  @ X_mu.T   # (n_vars, n_vars)
        J  += Xp_mu @ X_mu.T   # (n_vars, n_vars)

    # ------------------------------------------------------------------
    # Step 3: Data-driven rank selection via SVD of H  (Appendix B)
    #
    # The singular values of H represent the energy captured by each
    # spatial direction across all trajectories. We keep enough modes
    # to explain `variance_threshold` of total singular-value mass.
    # ------------------------------------------------------------------
    U, s, Vt = np.linalg.svd(H, full_matrices=False)   # s shape (n_vars,)

    cumulative_variance = np.cumsum(s) / np.sum(s)
    r = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    r = max(r, 1)

    print(f"[MTDMD] Chosen rank r = {r}  "
          f"(explains {cumulative_variance[r-1]*100:.1f}% of variance, "
          f"threshold = {variance_threshold*100:.1f}%)")

    # Truncated basis  (Appendix B, Eq. B2)
    Ur  = U[:, :r]    # (n_vars, r)  — spatial basis
    Sr  = np.diag(s[:r])
    Vtr = Vt[:r, :]   # (r, n_vars)

    # Reduced pseudo-inverse of H:  H⁺ ≈ Ur Sr⁻¹ Vtr
    Sr_inv = np.diag(1.0 / s[:r])
    H_pinv_r = Vtr.T @ Sr_inv @ Ur.T   # (n_vars, n_vars), but rank-r

    # ------------------------------------------------------------------
    # Step 4: Full MTDMD operator  A* = J · H⁺  (Eq. 45, B=0)
    #         then project to reduced space:  Ã = Urᵀ A* Ur  ∈ ℝ^{r×r}
    # ------------------------------------------------------------------
    A_full = J @ H_pinv_r          # (n_vars, n_vars)
    A_red  = Ur.T @ A_full @ Ur    # (r, r)  — reduced DMD operator

    # ------------------------------------------------------------------
    # Step 5: Eigendecomposition of the reduced operator
    # ------------------------------------------------------------------
    eigenvalues, W = np.linalg.eig(A_red)   # W columns are eigenvectors (r,r)

    # Lift eigenvectors back to sensor space → DMD modes  (n_vars, r)
    modes = Ur @ W   # (n_vars, r)

    # ------------------------------------------------------------------
    # Step 6: Reconstruct trajectories and compute mode amplitudes
    # ------------------------------------------------------------------
    reconstructions  = np.zeros((N_expr, n_ts, n_vars))
    mode_amplitudes  = []   # list of (r, n_ts) arrays

    for mu in range(N_expr):
        data_mu = experiment_timeseries[mu]   # (n_ts, n_vars)

        # Project initial condition onto reduced basis
        b0 = np.linalg.lstsq(modes, data_mu[0], rcond=None)[0]  # (r,)

        # Evolve in mode space:  b_{t+1} = diag(eigenvalues) · b_t
        # Since A_red = W diag(λ) W⁻¹,  b_t = W diag(λ)^t W⁻¹ b0
        W_inv = np.linalg.pinv(W)
        alpha0 = W_inv @ (Ur.T @ data_mu[0])   # (r,) initial amplitudes

        # Build temporal evolution for all n_ts steps
        t_steps = np.arange(n_ts)
        # lambda^t for each eigenvalue, shape (r, n_ts)
        lambda_t = np.array([eigenvalues ** t for t in t_steps]).T  # (r, n_ts)
        amplitudes = alpha0[:, np.newaxis] * lambda_t               # (r, n_ts)

        # Reconstruct in sensor space:  x̂(t) = modes @ W @ amplitudes[:, t]
        # = Ur @ W @ amplitudes[:, t]  but modes = Ur @ W already
        x_reconstructed = (modes @ amplitudes).T.real   # (n_ts, n_vars)

        reconstructions[mu] = x_reconstructed
        mode_amplitudes.append(amplitudes)

    mean_reconstruction = reconstructions.mean(axis=0)   # (n_ts, n_vars)

    return {
        "A" : A_red,
        "eigenvalues" : eigenvalues,
        "modes" : modes,
        "basis" : Ur,                    # (n_vars, r) real orthogonal SVD basis
        "rank" : r,
        "singular_values" : s,
        "reconstructions" : reconstructions,
        "mean_reconstruction" : mean_reconstruction,
        "mode_amplitudes" : mode_amplitudes,
    }


# ---------------------------------------------------------------------------
# Quick diagnostics / plotting helper  (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def plot_results(results: dict, experiment_timeseries: np.ndarray,
                 sensor_idx: int = 0):
    """
    Plot:
      1. Singular value spectrum with the rank cut-off marked.
      2. DMD eigenvalues in the complex plane.
      3. Original vs reconstructed signal for `sensor_idx` across experiments.
      4. Mean reconstruction vs per-experiment mean.
    """
    N, T, d = experiment_timeseries.shape
    r       = results["rank"]
    s       = results["singular_values"]
    eigs    = results["eigenvalues"]
    recs    = results["reconstructions"]
    mean_r  = results["mean_reconstruction"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("MTDMD Results", fontsize=14, fontweight="bold")

    # 1. Singular value spectrum
    ax = axes[0, 0]
    ax.semilogy(s / s.sum(), "o-", markersize=4, label="Normalised σ")
    ax.axvline(r - 1, color="red", linestyle="--",
               label=f"Rank cut-off r={r}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Normalised singular value (log)")
    ax.set_title("Hessian singular value spectrum")
    ax.legend()

    # 2. DMD eigenvalues in complex plane
    ax = axes[0, 1]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.5,
            label="Unit circle")
    ax.scatter(eigs.real, eigs.imag, s=60, zorder=5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("DMD eigenvalues")
    ax.set_aspect("equal")
    ax.legend()

    # 3. Original vs reconstructed — chosen sensor
    ax = axes[1, 0]
    t  = np.arange(T)
    for mu in range(N):
        ax.plot(t, experiment_timeseries[mu, :, sensor_idx],
                alpha=0.35, color="steelblue", linewidth=0.8)
        ax.plot(t, recs[mu, :, sensor_idx],
                alpha=0.6,  color="tomato",    linewidth=0.8,
                linestyle="--")
    ax.plot([], [], color="steelblue", label="Original")
    ax.plot([], [], color="tomato",    linestyle="--", label="Reconstructed")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sensor value")
    ax.set_title(f"Sensor {sensor_idx}: original vs reconstructed")
    ax.legend()

    # 4. Mean reconstruction vs raw mean
    ax = axes[1, 1]
    raw_mean = experiment_timeseries[:, :, sensor_idx].mean(axis=0)
    ax.plot(t, raw_mean,                    label="Raw mean",          linewidth=1.2)
    ax.plot(t, mean_r[:, sensor_idx],       label="MTDMD mean recon.", linewidth=1.2,
            linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sensor value")
    ax.set_title(f"Sensor {sensor_idx}: mean trajectories")
    ax.legend()

    plt.tight_layout()
    plt.show()
