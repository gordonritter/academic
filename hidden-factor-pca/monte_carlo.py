"""
Monte Carlo calibration of hidden-factor recovery via residual PCA.

Reproduces the figures in "Hidden Factors in Portfolio Risk Models: A
Finite-Sample Approach to Residual PCA" (Kolm & Ritter). Each figure plots,
for a fixed sample length T and universe size N, the result of a Monte Carlo
signal sweep of the standardized spiked-covariance procedure of Sections 3-4.

For each panel a single loading vector beta is drawn, the hidden-factor
volatility sigma_f is swept over a fixed grid, and at each grid point the
residual panel is simulated, standardized, and its leading principal component
recorded:

    x  = theta/sqrt(gamma) : spike strength in detection-threshold units; the
                             BBP phase transition sits at 1 for every N
    y1 = correl            : absolute alignment of leading PC, truth  (eq. 4.10)
    y2 = theta_hat/sqrt(gamma) : the estimate on the same threshold scale (eq. 3.27)

The x-axis is dimensionless and N-invariant, with the alignment knee anchored
near 1, so panels at different N are directly comparable.

Each plotted point may average over n_mc independent Monte Carlo draws (fresh
loading vector and panel per draw); n_mc=1 records a single draw per point.

The model and all formulas are taken directly from the paper; equation numbers
in the comments refer to its labeled equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationConfig:
    """Parameters for one Monte Carlo calibration run (one figure panel)."""

    N: int                              # number of assets (universe size)
    T: int = 300                        # number of time periods
    n_reps: int = 200                   # grid points (plotted points per panel)
    n_mc: int = 1                       # Monte Carlo draws averaged per grid point
    seed: int | None = 0                # RNG seed for reproducibility

    # --- Signal grid -------------------------------------------------------- #
    # The sweep is specified in detection-threshold units x = theta / sqrt(gamma),
    # where gamma = N/(T-1) and theta = sigma_f^2 * b'D^-1 b. The spiked-model
    # phase transition (BBP) sits at x = 1 for every N, so the x-axis is
    # dimensionless and N-invariant. For each target x the factor volatility is
    # set to realize theta = x * sqrt(gamma) on average (theta ~= sigma_f^2 *
    # tr(D^-1)), so sigma_f = sqrt(theta / tr(D^-1)).
    x_min: float = 0.0                  # smallest theta/sqrt(gamma) on the sweep
    x_max: float = 15.0                  # largest theta/sqrt(gamma) on the sweep

    # --- Idiosyncratic-variance profile D = diag(sigma_i^2) ----------------- #
    # Per-name residual volatilities. Absent the manager's real estimates we use
    # a lognormal volatility profile shaped to match the empirical US-equity
    # distribution: median ~0.025, bulk spanning ~0.005 to ~0.15 (right-skewed).
    # Override the whole profile via `vols`.
    vols: np.ndarray | None = None      # per-name residual volatilities (length N)
    vol_median: float = 0.025           # target median (lognormal median = exp(mu))
    vol_min: float = 0.005              # lower clip bound
    vol_max: float = 0.15               # upper clip bound
    vol_log_std: float = 0.9            # log-volatility dispersion (sets the spread)

    def aspect_ratio(self) -> float:
        """Aspect ratio gamma = N / T_eff, with T_eff = T - 1 (centered data)."""
        return self.N / (self.T - 1)


def build_volatilities(cfg: CalibrationConfig, rng: np.random.Generator) -> np.ndarray:
    """Return the fixed per-name residual volatility profile (length N).

    These are standard deviations sigma_i; the idiosyncratic variance profile is
    D = diag(sigma_i^2). Held fixed across the sweep, matching the paper's
    externally estimated D_hat (eq. 4.1).
    """
    if cfg.vols is not None:
        v = np.asarray(cfg.vols, dtype=float)
        if v.shape != (cfg.N,):
            raise ValueError(f"vols must have shape ({cfg.N},), got {v.shape}")
        return v
    # Lognormal with median pinned at vol_median (median = exp(mu)), then clipped
    # to [vol_min, vol_max]. Shaped to resemble the empirical US-equity profile.
    mu = np.log(cfg.vol_median)
    v = np.exp(rng.normal(mu, cfg.vol_log_std, size=cfg.N))
    return np.clip(v, cfg.vol_min, cfg.vol_max)


def load_vols(path: str) -> np.ndarray:
    """Load a residual-volatility array from a Java Arrays.toString() dump.

    The file is a comma-separated list of doubles wrapped in square brackets,
    e.g. "[0.0125, 0.0114, ...]". Returns a 1-D float array.
    """
    with open(path) as fh:
        text = fh.read().strip()
    text = text.strip("[]")
    return np.array([float(tok) for tok in text.split(",") if tok.strip()])


def theta_hat_from_lambda(lambda1: float, gamma: float) -> float:
    """Invert the outlier-eigenvalue map to estimate spike strength (eq. 3.27).

    Valid only above the noise edge, lambda1 >= (1 + sqrt(gamma))^2; below it the
    square-root argument is negative and theta_hat is undefined (returns NaN).
    """
    disc = (lambda1 - (1.0 + gamma)) ** 2 - 4.0 * gamma
    if disc < 0.0:
        return np.nan
    return 0.5 * ((lambda1 - (1.0 + gamma)) + np.sqrt(disc))


def threshold_units(theta: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Spike strength in detection-threshold units, x = theta / sqrt(gamma).

    gamma = N/(T-1) is the aspect ratio. The spiked-model (BBP) phase transition
    sits at x = 1 for every N, so this is the dimensionless, N-invariant scale on
    which the alignment knee is anchored near the threshold. Applying the same
    map to theta_hat gives the estimate on the same scale.
    """
    return theta / np.sqrt(gamma)


@dataclass
class CalibrationResult:
    """Per-grid-point arrays produced by `run_calibration`."""

    strength: np.ndarray        # mean true standardized spike strength theta (x-axis)
    correl: np.ndarray          # mean absolute alignment a (left y-axis)
    theta_hat: np.ndarray       # mean estimated spike strength theta_hat (right y-axis)
    cfg: CalibrationConfig = field(repr=False)


def _draw_leading_pc(
    cfg: CalibrationConfig,
    sigma_f: float,
    vols: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """One standardized PCA draw at factor volatility sigma_f.

    Draws a loading vector beta ~ N(0, I_N), simulates the residual panel,
    standardizes it, and returns (theta, absolute alignment, theta_hat) for the
    leading standardized principal component, where theta = sigma_f^2 * b'D^-1 b
    is the realized standardized spike strength for this draw (eq. 3.x).
    """
    D_inv = 1.0 / (vols ** 2)
    D_inv_sqrt = 1.0 / vols
    gamma = cfg.aspect_ratio()

    beta = rng.standard_normal(cfg.N)

    # Realized standardized spike strength theta = sigma_f^2 * beta' D^-1 beta.
    theta = sigma_f ** 2 * float(beta @ (D_inv * beta))

    # Standardized true direction D^-1/2 beta / ||.|| (eq. 4.10 target).
    b_std = D_inv_sqrt * beta
    b_std /= np.linalg.norm(b_std)

    # Residual panel R (T x N): idiosyncratic + hidden-factor part (eq. 4.9).
    r = rng.standard_normal((cfg.T, cfg.N)) * vols      # r_{i,t} ~ N(0, sigma_i^2)
    f = rng.standard_normal(cfg.T) * sigma_f            # f_t ~ N(0, sigma_f^2)
    R = r + np.outer(f, beta)

    # Center over time, scale by 1/sqrt(T-1) (eq. 3.5), whiten columns (eq. 3.9).
    X = (R - R.mean(axis=0, keepdims=True)) / np.sqrt(cfg.T - 1)
    X *= D_inv_sqrt

    # Leading eigenpair of X' X via the thin SVD of X: right singular vectors are
    # eigenvectors, squared singular values are eigenvalues.
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambda1 = float(s[0] ** 2)
    v1 = Vt[0]

    alignment = abs(float(v1 @ b_std))                  # eq. 4.10
    theta_hat = theta_hat_from_lambda(lambda1, gamma)   # eq. 3.27
    return theta, alignment, theta_hat


def run_calibration(cfg: CalibrationConfig) -> CalibrationResult:
    """Run the standardized Monte Carlo signal sweep for one (N, T) panel.

    The target spike strength in threshold units x = theta/sqrt(gamma) is swept
    linearly over [x_min, x_max]; for each x the factor volatility sigma_f is
    chosen so the spike strength realizes it on average (theta = x*sqrt(gamma),
    sigma_f = sqrt(theta / tr(D^-1))). At each grid point, `n_mc` independent
    draws (each with a fresh loading vector and panel) are averaged into a single
    plotted point:

      strength (x)  = mean true theta = sigma_f^2 * b'D^-1 b      (truth)
      correl   (y1) = mean absolute alignment                     (eq. 4.10)
      theta_hat(y2) = mean estimated theta_hat                     (eq. 3.27)

    The figure converts strength and theta_hat to threshold units via
    threshold_units(.), so both axes are dimensionless and N-invariant, the knee
    is anchored near x = 1 (BBP), and the slope-1 relationship of the estimate
    is preserved. For theta_hat only draws above the noise edge contribute (a
    point is NaN only if every draw falls below the edge).
    """
    rng = np.random.default_rng(cfg.seed)
    vols = build_volatilities(cfg, rng)
    trace_D_inv = float((1.0 / vols ** 2).sum())        # E[b'D^-1 b] = tr(D^-1)
    sqrt_gamma = np.sqrt(cfg.aspect_ratio())

    # Map the target threshold-unit grid to factor volatilities.
    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.n_reps)
    theta_targets = x_grid * sqrt_gamma
    sigma_grid = np.sqrt(theta_targets / trace_D_inv)

    strength = np.empty(cfg.n_reps)
    correl = np.empty(cfg.n_reps)
    theta_hat = np.empty(cfg.n_reps)

    for k, sigma_f in enumerate(sigma_grid):
        thetas = np.empty(cfg.n_mc)
        aligns = np.empty(cfg.n_mc)
        theta_hats = np.empty(cfg.n_mc)
        for j in range(cfg.n_mc):
            thetas[j], aligns[j], theta_hats[j] = _draw_leading_pc(
                cfg, sigma_f, vols, rng
            )

        strength[k] = thetas.mean()                     # true standardized strength
        correl[k] = aligns.mean()
        # Average only the defined theta_hat draws; NaN if none are defined.
        defined = theta_hats[~np.isnan(theta_hats)]
        theta_hat[k] = defined.mean() if defined.size else np.nan

    return CalibrationResult(
        strength=strength, correl=correl, theta_hat=theta_hat, cfg=cfg
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _apply_latex_style() -> None:
    """Set Matplotlib rcParams to a Computer Modern / LaTeX-style appearance.

    Uses mathtext with the Computer Modern font set and a serif text family,
    which gives a LaTeX look without requiring a system LaTeX installation.
    """
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "CMU Serif", "Computer Modern Roman", "DejaVu Serif",
    ]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["axes.formatter.use_mathtext"] = True


def make_figure(result: CalibrationResult):
    """Build the dual-axis scatter figure for one panel.

    The x-axis is the spike strength in detection-threshold units,
    x = theta / sqrt(gamma), so the BBP phase transition sits at x = 1 for every
    N. Black markers show the absolute alignment `a` (left axis, 0..1); the
    knee is anchored near x = 1. Red markers show the estimate theta_hat in the
    same units (right axis), with the dashed y = x line as the slope-1 reference.
    A vertical line marks the x = 1 threshold.

    Returns the matplotlib Figure; the caller is responsible for saving/closing.
    """
    import matplotlib.pyplot as plt

    _apply_latex_style()

    cfg = result.cfg
    gamma = cfg.aspect_ratio()
    x = threshold_units(result.strength, gamma)
    x_hat = threshold_units(result.theta_hat, gamma)

    fig, ax_left = plt.subplots(figsize=(8, 6))
    ax_right = ax_left.twinx()

    left = ax_left.scatter(
        x, result.correl, s=12, c="black", label=r"Absolute Alignment $a$"
    )
    right = ax_right.scatter(
        x, x_hat, s=12, c="red",
        label=r"Implied Strength Estimate $\widehat{\theta}/\sqrt{\gamma}$",
    )

    # Both axes are in the same threshold units; slope-1 diagonal and the BBP
    # threshold marker at x = 1.
    hi = cfg.x_max
    diag = ax_right.plot(
        [0.0, hi], [0.0, hi], ls="--", lw=1, c="red", alpha=0.5,
        label="Slope 1",
    )[0]
    thresh = ax_left.axvline(
        1.0, ls=":", lw=1, c="gray", label=r"BBP Threshold ($x = 1$)"
    )
    ax_left.set_xlim(0.0, hi)
    ax_right.set_ylim(0.0, hi)

    ax_left.set_xlabel(r"Spike Strength $\theta/\sqrt{\gamma}$")
    ax_left.set_ylabel(r"Absolute Alignment $a$")
    ax_right.set_ylabel(
        r"Implied Strength Estimate $\widehat{\theta}/\sqrt{\gamma}$"
    )
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(handles=[left, right, diag, thresh], loc="center right")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(
    universe_sizes: tuple[int, ...] = (250, 500, 1000, 1500, 2000, 3000),
    T: int = 300,
    n_reps: int = 200,
    n_mc: int = 1,
    out_dir: str = ".",
    vol_file_pattern: str | None = None,
) -> None:
    """Generate one standardized calibration figure per universe size as PNG.

    If `vol_file_pattern` is given, it is formatted with each N to locate a vol
    file (e.g. "vols-{N}.txt") whose contents become that panel's volatility
    profile; the loaded length must equal N. Otherwise a synthetic lognormal
    profile is used.
    """
    import os

    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for N in universe_sizes:
        vols = None
        if vol_file_pattern is not None:
            vols = load_vols(vol_file_pattern.format(N=N))
            if vols.shape != (N,):
                raise ValueError(
                    f"{vol_file_pattern.format(N=N)} has {vols.size} vols, expected {N}"
                )
        cfg = CalibrationConfig(N=N, T=T, n_reps=n_reps, n_mc=n_mc, seed=N, vols=vols)
        result = run_calibration(cfg)
        fig = make_figure(result)
        path = os.path.join(out_dir, f"correl-N{N}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"N={N}: wrote {path}")

if __name__ == "__main__":
    main(vol_file_pattern = "vols-{N}.txt", n_mc = 100)
