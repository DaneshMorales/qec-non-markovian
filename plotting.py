"""
Visualization utilities for logical randomized-benchmarking results.

The comparison figure contains:

    (a) Mean survival probabilities with exponential RB fits.
        Error bars show the standard error of the mean when available.

    (b) Cross-sequence standard deviations.

    (c) Non-Markovian survival data minus the fitted Markovian-reference
        curve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .fitting import rb_model


# ---------------------------------------------------------------------------
# Validation and formatting helpers
# ---------------------------------------------------------------------------

def _get_1d_array(
    results: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    """Extract a finite one-dimensional numerical array."""
    if key not in results:
        raise KeyError(f"Results dictionary is missing key {key!r}")

    array = np.asarray(results[key], dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"results[{key!r}] must be one-dimensional, "
            f"got shape {array.shape}"
        )

    if array.size == 0:
        raise ValueError(
            f"results[{key!r}] must not be empty"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"results[{key!r}] contains nonfinite entries"
        )

    return array


def _extract_result_arrays(
    results: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return sequence lengths, survival means, and cross-sequence standard
    deviations.
    """
    lengths = _get_1d_array(
        results,
        "sequence_lengths",
    )

    means = _get_1d_array(
        results,
        "survival_means",
    )

    standard_deviations = _get_1d_array(
        results,
        "survival_stds",
    )

    if not (
        lengths.shape
        == means.shape
        == standard_deviations.shape
    ):
        raise ValueError(
            "sequence_lengths, survival_means, and survival_stds "
            "must have identical shapes"
        )

    if np.any(lengths < 0):
        raise ValueError(
            "Sequence lengths must be nonnegative"
        )

    if np.any(means < -1e-8) or np.any(means > 1.0 + 1e-8):
        raise ValueError(
            "Survival means must lie in [0, 1]"
        )

    if np.any(standard_deviations < 0):
        raise ValueError(
            "Survival standard deviations must be nonnegative"
        )

    means = np.clip(means, 0.0, 1.0)

    return lengths, means, standard_deviations


def _mean_error_bars(
    results: Mapping[str, Any],
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    """
    Return standard errors for plotting mean survival probabilities.

    Priority:
        1. survival_sems supplied by run_logical_rb()
        2. SEM calculated from all_seq_means
        3. no error bars
    """
    if "survival_sems" in results:
        sems = np.asarray(
            results["survival_sems"],
            dtype=float,
        )

        if sems.shape != expected_shape:
            raise ValueError(
                "results['survival_sems'] must have shape "
                f"{expected_shape}, got {sems.shape}"
            )

        if not np.all(np.isfinite(sems)):
            raise ValueError(
                "results['survival_sems'] contains nonfinite entries"
            )

        if np.any(sems < 0):
            raise ValueError(
                "Standard errors must be nonnegative"
            )

        return sems

    if "all_seq_means" in results:
        all_seq_means = np.asarray(
            results["all_seq_means"],
            dtype=float,
        )

        if all_seq_means.ndim != 2:
            raise ValueError(
                "results['all_seq_means'] must be two-dimensional"
            )

        if all_seq_means.shape[0] != expected_shape[0]:
            raise ValueError(
                "The first dimension of all_seq_means must match "
                "the number of sequence lengths"
            )

        if not np.all(np.isfinite(all_seq_means)):
            raise ValueError(
                "results['all_seq_means'] contains nonfinite entries"
            )

        number_of_sequences = all_seq_means.shape[1]

        if number_of_sequences >= 2:
            return (
                np.std(
                    all_seq_means,
                    axis=1,
                    ddof=1,
                )
                / np.sqrt(number_of_sequences)
            )

    return None


def _sorted(
    lengths: np.ndarray,
    *values: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Sort one or more arrays according to increasing sequence length."""
    order = np.argsort(lengths)

    return (
        lengths[order],
        *(value[order] for value in values),
    )


def _fit_is_valid(
    fit: Mapping[str, Any],
) -> bool:
    """Return whether a fit contains finite A, p, and B parameters."""
    try:
        parameters = np.asarray(
            [
                fit["A"],
                fit["p"],
                fit["B"],
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return False

    return bool(np.all(np.isfinite(parameters)))


def _fit_label(
    prefix: str,
    fit: Mapping[str, Any],
) -> str:
    """Construct a fit label with an uncertainty when available."""
    p = float(fit["p"])

    try:
        p_error = float(fit["p_err"])
    except (KeyError, TypeError, ValueError):
        p_error = float("nan")

    if np.isfinite(p_error):
        return f"{prefix} fit: p={p:.4f} ± {p_error:.4f}"

    return f"{prefix} fit: p={p:.4f}"


def _save_figure(
    figure: plt.Figure,
    save_path: str | PathLike[str] | None,
) -> None:
    """Save a figure when a path is supplied."""
    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )


# ---------------------------------------------------------------------------
# Non-Markovian versus Markovian comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    results_nm: Mapping[str, Any],
    results_mk: Mapping[str, Any],
    fit_nm: Mapping[str, Any],
    fit_mk: Mapping[str, Any],
    diag: Mapping[str, Any] | None = None,
    title: str = (
        "Logical RB: Non-Markovian vs Markovian reference"
    ),
    save_path: str | PathLike[str] | None = None,
) -> plt.Figure:
    """
    Plot a three-panel comparison.

    Panels
    ------
    (a)
        Mean survival probabilities and exponential RB fits. Error bars
        represent standard errors of the means.

    (b)
        Cross-sequence standard deviations.

    (c)
        Non-Markovian means minus the fitted Markovian-reference curve.
    """
    (
        lengths_nm,
        means_nm,
        standard_deviations_nm,
    ) = _extract_result_arrays(results_nm)

    (
        lengths_mk,
        means_mk,
        standard_deviations_mk,
    ) = _extract_result_arrays(results_mk)

    sems_nm = _mean_error_bars(
        results_nm,
        lengths_nm.shape,
    )

    sems_mk = _mean_error_bars(
        results_mk,
        lengths_mk.shape,
    )

    (
        lengths_nm,
        means_nm,
        standard_deviations_nm,
    ) = _sorted(
        lengths_nm,
        means_nm,
        standard_deviations_nm,
    )

    (
        lengths_mk,
        means_mk,
        standard_deviations_mk,
    ) = _sorted(
        lengths_mk,
        means_mk,
        standard_deviations_mk,
    )

    if sems_nm is not None:
        _, sems_nm = _sorted(
            _get_1d_array(results_nm, "sequence_lengths"),
            sems_nm,
        )

    if sems_mk is not None:
        _, sems_mk = _sorted(
            _get_1d_array(results_mk, "sequence_lengths"),
            sems_mk,
        )

    minimum_length = min(
        float(np.min(lengths_nm)),
        float(np.min(lengths_mk)),
    )

    maximum_length = max(
        float(np.max(lengths_nm)),
        float(np.max(lengths_mk)),
    )

    if np.isclose(minimum_length, maximum_length):
        fit_lengths = np.array(
            [minimum_length],
            dtype=float,
        )
    else:
        fit_lengths = np.linspace(
            minimum_length,
            maximum_length,
            400,
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    survival_axis, spread_axis, residual_axis = axes

    # ------------------------------------------------------------------
    # Panel (a): survival means and fitted curves
    # ------------------------------------------------------------------

    survival_axis.errorbar(
        lengths_nm,
        means_nm,
        yerr=sems_nm,
        fmt="o",
        capsize=4,
        label="Non-Markovian",
        zorder=3,
    )

    survival_axis.errorbar(
        lengths_mk,
        means_mk,
        yerr=sems_mk,
        fmt="s",
        capsize=4,
        alpha=0.75,
        label="Markovian reference",
        zorder=3,
    )

    if _fit_is_valid(fit_nm):
        survival_axis.plot(
            fit_lengths,
            rb_model(
                fit_lengths,
                float(fit_nm["A"]),
                float(fit_nm["p"]),
                float(fit_nm["B"]),
            ),
            "-",
            linewidth=1.5,
            label=_fit_label(
                "NM",
                fit_nm,
            ),
        )

    if _fit_is_valid(fit_mk):
        survival_axis.plot(
            fit_lengths,
            rb_model(
                fit_lengths,
                float(fit_mk["A"]),
                float(fit_mk["p"]),
                float(fit_mk["B"]),
            ),
            "--",
            linewidth=1.5,
            label=_fit_label(
                "MK",
                fit_mk,
            ),
        )

    survival_axis.set_xlabel("Sequence length $m$")
    survival_axis.set_ylabel(
        r"Mean survival probability $\bar{P}(m)$"
    )
    survival_axis.set_title("(a) Survival curves")
    survival_axis.set_ylim(-0.05, 1.05)
    survival_axis.grid(True, alpha=0.3)
    survival_axis.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel (b): cross-sequence standard deviation
    # ------------------------------------------------------------------

    spread_axis.plot(
        lengths_nm,
        standard_deviations_nm,
        "o-",
        label="Non-Markovian",
    )

    spread_axis.plot(
        lengths_mk,
        standard_deviations_mk,
        "s--",
        label="Markovian reference",
    )

    spread_axis.set_xlabel("Sequence length $m$")
    spread_axis.set_ylabel(
        "Cross-sequence standard deviation"
    )
    spread_axis.set_title(
        "(b) Cross-sequence spread"
    )
    spread_axis.grid(True, alpha=0.3)
    spread_axis.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel (c): NM data minus fitted MK curve
    # ------------------------------------------------------------------

    residual_axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    if diag is None:
        residual_axis.text(
            0.5,
            0.5,
            "No diagnostic data",
            horizontalalignment="center",
            verticalalignment="center",
            transform=residual_axis.transAxes,
        )

        residual_axis.set_title(
            "(c) NM − MK-fit residuals"
        )

    else:
        if "residuals_nm_vs_mk" not in diag:
            raise KeyError(
                "Diagnostic dictionary must contain "
                "'residuals_nm_vs_mk'"
            )

        residuals = np.asarray(
            diag["residuals_nm_vs_mk"],
            dtype=float,
        )

        if residuals.shape != lengths_nm.shape:
            raise ValueError(
                "diag['residuals_nm_vs_mk'] must have shape "
                f"{lengths_nm.shape}, got {residuals.shape}"
            )

        if not np.all(np.isfinite(residuals)):
            raise ValueError(
                "diag['residuals_nm_vs_mk'] contains "
                "nonfinite entries"
            )

        if "sequence_lengths" in diag:
            diagnostic_lengths = np.asarray(
                diag["sequence_lengths"],
                dtype=float,
            )

            if diagnostic_lengths.shape != residuals.shape:
                raise ValueError(
                    "diag['sequence_lengths'] and residuals must "
                    "have identical shapes"
                )
        else:
            diagnostic_lengths = np.asarray(
                results_nm["sequence_lengths"],
                dtype=float,
            )

        diagnostic_lengths, residuals = _sorted(
            diagnostic_lengths,
            residuals,
        )

        residual_axis.plot(
            diagnostic_lengths,
            residuals,
            "o-",
            label="NM data − MK fit",
        )

        residual_axis.fill_between(
            diagnostic_lengths,
            0.0,
            residuals,
            alpha=0.2,
        )

        try:
            maximum_deviation = float(
                diag["max_nm_mk_deviation"]
            )
        except (KeyError, TypeError, ValueError):
            maximum_deviation = float(
                np.max(np.abs(residuals))
            )

        residual_axis.set_title(
            "(c) NM − MK-fit residuals\n"
            f"max |deviation| = {maximum_deviation:.4f}"
        )

        residual_axis.legend(fontsize=8)

    residual_axis.set_xlabel("Sequence length $m$")
    residual_axis.set_ylabel(
        r"$\bar{P}_{\mathrm{NM}}(m)"
        r"-\bar{P}_{\mathrm{MK,fit}}(m)$"
    )
    residual_axis.grid(True, alpha=0.3)

    figure.suptitle(
        title,
        fontsize=12,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
    )

    _save_figure(
        figure,
        save_path,
    )

    return figure


# ---------------------------------------------------------------------------
# Markovianization diagnostics
# ---------------------------------------------------------------------------

def plot_markovianization_diagnostics(
    results_nm: Mapping[str, Any],
    results_mk: Mapping[str, Any],
    fit_nm: Mapping[str, Any],
    fit_mk: Mapping[str, Any],
    title: str = "QEC Markovianization diagnostics",
    save_path: str | PathLike[str] | None = None,
) -> plt.Figure:
    """
    Four-panel figure for testing whether QEC has Markovianized the noise.

    Each panel captures a distinct signature of non-Markovian behaviour:

    (a) Semi-log survival (mean − B_fit) vs m.  A straight line confirms
        geometric (Markovian) decay; curvature signals non-exponential
        behaviour driven by environment memory.

    (b) Effective per-step fidelity p_eff(m) between consecutive sequence
        lengths: p_eff = [(P(m₂)−B)/(P(m₁)−B)]^{1/(m₂−m₁)}.
        Markovian noise gives a flat line at p_fit; non-Markovian noise
        shows a trend (decay rate speeds up or slows with m).

    (c) Cross-sequence variance ratio σ²_NM(m)/σ²_MK(m).  A ratio
        significantly above 1 indicates that the non-Markovian run has
        excess sequence-to-sequence spread — a sign that environment memory
        biases some Clifford sequences more than others.

    (d) Signed survival gap P̄_NM(m) − P̄_MK(m).  Exactly zero at every m
        means QEC fully Markovianized the noise.  A positive gap means the
        non-Markovian run decays *slower* (temporal correlations cancel
        errors); a negative gap means it decays *faster*.

    Parameters
    ----------
    results_nm, results_mk : dict
        Output of ``run_logical_rb()`` for the non-Markovian and Markovian
        reference runs.
    fit_nm, fit_mk : dict
        Output of ``fit_rb_curve()`` for the corresponding runs.
    title : str
        Figure suptitle.
    save_path : path-like or None
        If provided, the figure is saved there before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lengths_nm, means_nm, stds_nm = _extract_result_arrays(results_nm)
    lengths_mk, means_mk, stds_mk = _extract_result_arrays(results_mk)
    lengths_nm, means_nm, stds_nm = _sorted(lengths_nm, means_nm, stds_nm)
    lengths_mk, means_mk, stds_mk = _sorted(lengths_mk, means_mk, stds_mk)

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    ax_log, ax_rate, ax_var, ax_gap = axes

    _C = {"NM": "steelblue", "MK": "darkorange"}

    # ------------------------------------------------------------------
    # Panel (a): semi-log survival (B-subtracted)
    # ------------------------------------------------------------------
    for lengths, means, fit, tag, fmt in [
        (lengths_nm, means_nm, fit_nm, "NM", "o-"),
        (lengths_mk, means_mk, fit_mk, "MK", "s--"),
    ]:
        B = float(fit["B"]) if _fit_is_valid(fit) else 0.5
        shifted = np.clip(means - B, 1e-10, None)
        ax_log.semilogy(
            lengths, shifted, fmt, color=_C[tag], markersize=5,
            label=f"{tag} data (B={B:.3f})",
        )
        if _fit_is_valid(fit):
            A_f = float(fit["A"])
            p_f = float(fit["p"])
            m_dense = np.linspace(float(lengths[0]), float(lengths[-1]), 300)
            ax_log.semilogy(
                m_dense, np.clip(A_f * p_f ** m_dense, 1e-10, None),
                "-", color=_C[tag], linewidth=1.0, alpha=0.5,
            )

    ax_log.set_xlabel("Sequence length $m$")
    ax_log.set_ylabel(r"$\bar{P}(m) - B_{\rm fit}$ (log scale)")
    ax_log.set_title("(a) Semi-log survival\nstraight line = geometric (Markovian) decay")
    ax_log.legend(fontsize=7)
    ax_log.grid(True, alpha=0.3, which="both")

    # ------------------------------------------------------------------
    # Panel (b): effective per-step fidelity
    # ------------------------------------------------------------------
    for lengths, means, fit, tag, fmt in [
        (lengths_nm, means_nm, fit_nm, "NM", "o-"),
        (lengths_mk, means_mk, fit_mk, "MK", "s--"),
    ]:
        if len(lengths) < 2 or not _fit_is_valid(fit):
            continue
        B   = float(fit["B"])
        p_f = float(fit["p"])
        mids, p_effs = [], []
        for i in range(len(lengths) - 1):
            m1, m2 = float(lengths[i]), float(lengths[i + 1])
            P1 = float(means[i]) - B
            P2 = float(means[i + 1]) - B
            dm = m2 - m1
            if dm > 0 and P1 > 1e-8 and P2 > 0:
                mids.append((m1 + m2) / 2.0)
                p_effs.append((P2 / P1) ** (1.0 / dm))
        if mids:
            ax_rate.plot(
                mids, p_effs, fmt, color=_C[tag], markersize=5,
                label=f"{tag} (p_fit={p_f:.3f})",
            )
            ax_rate.axhline(
                p_f, linestyle=":", color=_C[tag], linewidth=0.9, alpha=0.7,
            )

    ax_rate.set_xlabel("Midpoint sequence length")
    ax_rate.set_ylabel(r"$p_{\rm eff}(m)$")
    ax_rate.set_title("(b) Per-step decay rate\nflat = stationary (Markovian)")
    ax_rate.legend(fontsize=7)
    ax_rate.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (c): variance ratio σ²_NM / σ²_MK
    # ------------------------------------------------------------------
    nm_std = dict(zip(lengths_nm.astype(int), stds_nm))
    mk_std = dict(zip(lengths_mk.astype(int), stds_mk))
    common_m = sorted(set(nm_std) & set(mk_std))

    ratio_m, ratio_v = [], []
    for m in common_m:
        s_mk = mk_std[m]
        if s_mk > 1e-12:
            ratio_m.append(m)
            ratio_v.append((nm_std[m] / s_mk) ** 2)

    if ratio_m:
        ax_var.plot(
            ratio_m, ratio_v, "o-", color="mediumpurple", markersize=5,
            label=r"$\sigma^2_{\rm NM}\,/\,\sigma^2_{\rm MK}$",
        )

    ax_var.axhline(1.0, linestyle="--", linewidth=0.9, color="gray", label="ratio = 1")
    ax_var.set_xlabel("Sequence length $m$")
    ax_var.set_ylabel(r"$\sigma^2_{\rm NM}\,/\,\sigma^2_{\rm MK}$")
    ax_var.set_title("(c) Variance ratio\n>1 = excess non-Markovian spread")
    ax_var.legend(fontsize=7)
    ax_var.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (d): signed NM − MK gap
    # ------------------------------------------------------------------
    nm_mean = dict(zip(lengths_nm.astype(int), means_nm))
    mk_mean = dict(zip(lengths_mk.astype(int), means_mk))
    gap_m   = sorted(set(nm_mean) & set(mk_mean))
    gaps    = [nm_mean[m] - mk_mean[m] for m in gap_m]

    if gap_m:
        ax_gap.plot(
            gap_m, gaps, "o-", color="darkgreen", markersize=5,
            label=r"$\bar{P}_{\rm NM} - \bar{P}_{\rm MK}$",
        )
        ax_gap.fill_between(gap_m, 0.0, gaps, alpha=0.2, color="darkgreen")

    ax_gap.axhline(0.0, linestyle="--", linewidth=0.9, color="gray",
                   label="zero (Markovianized)")
    ax_gap.set_xlabel("Sequence length $m$")
    ax_gap.set_ylabel(r"$\bar{P}_{\rm NM}(m) - \bar{P}_{\rm MK}(m)$")
    ax_gap.set_title("(d) Signed NM − MK gap\nzero = fully Markovianized")
    ax_gap.legend(fontsize=7)
    ax_gap.grid(True, alpha=0.3)

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save_figure(figure, save_path)
    return figure


# ---------------------------------------------------------------------------
# Correction-frequency sweep
# ---------------------------------------------------------------------------

def plot_correction_frequency_sweep(
    sweep_results: Sequence[Mapping[str, Any]],
    sweep_label: str = "correction frequency",
    save_path: str | PathLike[str] | None = None,
) -> plt.Figure:
    """
    Plot several logical-RB runs from a correction-frequency sweep.

    Each result dictionary should come from run_logical_rb() and may include
    a human-readable ``label`` entry.

    The first panel shows mean survival probabilities with SEM error bars.
    The second panel shows cross-sequence standard deviations.
    """
    if len(sweep_results) == 0:
        raise ValueError(
            "sweep_results must contain at least one result"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    survival_axis, spread_axis = axes

    color_map = plt.colormaps["viridis"]
    number_of_runs = len(sweep_results)

    for index, result in enumerate(sweep_results):
        (
            lengths,
            means,
            standard_deviations,
        ) = _extract_result_arrays(result)

        sems = _mean_error_bars(
            result,
            lengths.shape,
        )

        order = np.argsort(lengths)

        lengths = lengths[order]
        means = means[order]
        standard_deviations = (
            standard_deviations[order]
        )

        if sems is not None:
            sems = sems[order]

        denominator = max(
            number_of_runs - 1,
            1,
        )

        color = color_map(
            index / denominator
        )

        label = str(
            result.get(
                "label",
                f"run {index + 1}",
            )
        )

        survival_axis.errorbar(
            lengths,
            means,
            yerr=sems,
            fmt="o-",
            capsize=3,
            color=color,
            label=label,
            alpha=0.85,
        )

        spread_axis.plot(
            lengths,
            standard_deviations,
            "o-",
            color=color,
            label=label,
            alpha=0.85,
        )

    survival_axis.set_xlabel(
        "Sequence length $m$"
    )
    survival_axis.set_ylabel(
        r"Mean survival probability $\bar{P}(m)$"
    )
    survival_axis.set_title(
        f"Survival vs {sweep_label}"
    )
    survival_axis.set_ylim(-0.05, 1.05)
    survival_axis.grid(True, alpha=0.3)
    survival_axis.legend(fontsize=7)

    spread_axis.set_xlabel(
        "Sequence length $m$"
    )
    spread_axis.set_ylabel(
        "Cross-sequence standard deviation"
    )
    spread_axis.set_title(
        f"Cross-sequence spread vs {sweep_label}"
    )
    spread_axis.grid(True, alpha=0.3)
    spread_axis.legend(fontsize=7)

    figure.tight_layout()

    _save_figure(
        figure,
        save_path,
    )

    return figure