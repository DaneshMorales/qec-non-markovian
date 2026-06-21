"""
RB curve fitting and non-Markovian diagnostics.

Fits

    P_bar(m) = A p**m + B

using nonlinear least squares.

When available, the fit is weighted using the standard error of the mean
survival probability at each sequence length. Cross-sequence standard
deviations are not used directly as uncertainties in the fitted means.

The diagnostics compare non-Markovian and Markovian-reference simulations
using:

    - deviations from exponential RB decay,
    - deviations from the Markovian-reference fit,
    - cross-sequence variance ratios,
    - upward survival-probability revivals,
    - fitted decay-parameter differences.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


# ---------------------------------------------------------------------------
# RB model
# ---------------------------------------------------------------------------

def rb_model(
    m: np.ndarray,
    A: float,
    p: float,
    B: float,
) -> np.ndarray:
    """
    Standard RB decay model

        P_bar(m) = A p**m + B.
    """
    m = np.asarray(m, dtype=float)
    return A * np.power(p, m) + B


# ---------------------------------------------------------------------------
# Validation and fitting helpers
# ---------------------------------------------------------------------------

def _get_1d_array(
    results: Mapping[str, Any],
    key: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    """Read and validate a one-dimensional numerical result array."""
    if key not in results:
        raise KeyError(f"Results dictionary is missing key {key!r}")

    array = np.asarray(results[key], dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"results[{key!r}] must be one-dimensional, "
            f"got shape {array.shape}"
        )

    if finite and not np.all(np.isfinite(array)):
        raise ValueError(
            f"results[{key!r}] contains nonfinite entries"
        )

    return array


def _regularize_sigma(
    sigma: np.ndarray,
) -> np.ndarray | None:
    """
    Replace isolated zero or invalid uncertainties with a conservative floor.

    If every uncertainty is zero or invalid, return None so the fit is
    performed without weighting.
    """
    sigma = np.asarray(sigma, dtype=float)

    valid_positive = (
        np.isfinite(sigma)
        & (sigma > 0.0)
    )

    positive_values = sigma[valid_positive]

    if positive_values.size == 0:
        return None

    # Avoid treating an empirically zero uncertainty as infinite precision.
    floor = max(
        1e-8,
        0.1 * float(np.median(positive_values)),
    )

    return np.where(
        valid_positive,
        np.maximum(sigma, floor),
        floor,
    )


def _extract_fit_sigma(
    results: Mapping[str, Any],
    number_of_lengths: int,
) -> tuple[np.ndarray | None, str]:
    """
    Obtain uncertainties for fitting mean survival probabilities.

    Priority:
        1. results["survival_sems"]
        2. SEM calculated from results["all_seq_means"]
        3. unweighted fitting

    The cross-sequence standard deviation is not itself the uncertainty of
    the cross-sequence mean, so results["survival_stds"] is not used directly.
    """
    if "survival_sems" in results:
        sems = np.asarray(
            results["survival_sems"],
            dtype=float,
        )

        if sems.shape != (number_of_lengths,):
            raise ValueError(
                "results['survival_sems'] must have shape "
                f"{(number_of_lengths,)}, got {sems.shape}"
            )

        sigma = _regularize_sigma(sems)

        if sigma is not None:
            return sigma, "survival_sems"

    if "all_seq_means" in results:
        all_seq_means = np.asarray(
            results["all_seq_means"],
            dtype=float,
        )

        if all_seq_means.ndim != 2:
            raise ValueError(
                "results['all_seq_means'] must be two-dimensional, "
                f"got shape {all_seq_means.shape}"
            )

        if all_seq_means.shape[0] != number_of_lengths:
            raise ValueError(
                "The first dimension of results['all_seq_means'] must "
                f"equal {number_of_lengths}, got "
                f"{all_seq_means.shape[0]}"
            )

        if not np.all(np.isfinite(all_seq_means)):
            raise ValueError(
                "results['all_seq_means'] contains nonfinite entries"
            )

        number_of_sequences = all_seq_means.shape[1]

        if number_of_sequences >= 2:
            sems = (
                np.std(
                    all_seq_means,
                    axis=1,
                    ddof=1,
                )
                / np.sqrt(number_of_sequences)
            )

            sigma = _regularize_sigma(sems)

            if sigma is not None:
                return sigma, "SEM from all_seq_means"

    return None, "unweighted"


def _initial_guess(
    sequence_lengths: np.ndarray,
    survival_means: np.ndarray,
) -> tuple[float, float, float]:
    """
    Construct a bounded initial guess (A0, p0, B0) for the RB fit.

    Heuristics
    ----------
    B0 — estimated from the mean of the last (largest-m) few points, where
         the decay p^m ≈ 0 and the curve has settled near its asymptote.
    A0 — estimated as P(m=0) − B0, i.e. the initial amplitude above the floor.
    p0 — fixed at 0.95 (a conservative starting decay rate for most codes).
    """
    order = np.argsort(sequence_lengths)
    sorted_means = survival_means[order]

    number_in_tail = min(3, len(sorted_means))

    B0 = float(
        np.mean(sorted_means[-number_in_tail:])
    )

    # Keep the starting point strictly inside the bounded region.
    B0 = float(np.clip(B0, 1e-6, 1.0 - 1e-6))

    A0 = float(sorted_means[0] - B0)
    A0 = float(np.clip(A0, -0.999, 0.999))

    if abs(A0) < 1e-4:
        # Flat curve: fall back to the total spread as a rough amplitude.
        spread = float(sorted_means[-1] - sorted_means[0])

        if spread > 1e-4:
            A0 = float(np.clip(spread, 1e-4, 0.999))
        else:
            A0 = 0.1

    p0 = 0.95

    return A0, p0, B0


def _parameter_errors(
    covariance: np.ndarray,
) -> np.ndarray:
    """Extract valid one-standard-deviation parameter errors."""
    covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (3, 3):
        return np.full(3, np.nan)

    diagonal = np.diag(covariance)

    errors = np.full(3, np.nan)
    valid = np.isfinite(diagonal) & (diagonal >= 0.0)
    errors[valid] = np.sqrt(diagonal[valid])

    return errors


def _fit_parameters_are_valid(
    fit: Mapping[str, Any],
) -> bool:
    """Return whether a fit dictionary contains finite A, p, and B."""
    try:
        parameters = np.asarray(
            [fit["A"], fit["p"], fit["B"]],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return False

    return bool(np.all(np.isfinite(parameters)))


def _maximum_absolute(
    values: np.ndarray,
) -> float:
    """Return the largest finite absolute value, or NaN."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not np.any(finite):
        return float("nan")

    return float(np.max(np.abs(values[finite])))


# ---------------------------------------------------------------------------
# RB fitting
# ---------------------------------------------------------------------------

def fit_rb_curve(
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Fit survival data to the standard RB model

        P_bar(m) = A p**m + B.

    The fit uses standard errors of the means (SEMs) when available.
    The weighting priority is: survival_sems > SEM from all_seq_means > unweighted.

    Parameters
    ----------
    results : dict
        Result dictionary returned by run_logical_rb().  Must contain at least
        ``sequence_lengths`` and ``survival_means``; ``survival_sems`` or
        ``all_seq_means`` are used for weighted fitting when present.

    Returns
    -------
    dict with keys
        A, p, B         — fitted RB model parameters (NaN on failure)
        A_err, p_err,
        B_err           — one-std-deviation parameter errors from covariance
        fit_curve       — model evaluated at the data sequence lengths
        residuals       — survival_means − fit_curve
        covariance      — 3×3 parameter covariance matrix
        rmse            — root-mean-square residual
        r_squared       — coefficient of determination
        chi_squared     — Σ((residual/sigma)²); NaN if unweighted
        reduced_chi_squared — chi_squared / (n_pts − 3); NaN if unweighted
        standardized_residuals — residuals/sigma; NaN if unweighted
        weighting       — string describing which sigma source was used
        success         — bool; False if the optimizer did not converge
        message         — human-readable fit status string
        warnings        — list of optimizer warning strings (empty if none)
    """
    sequence_lengths = _get_1d_array(
        results,
        "sequence_lengths",
    )

    survival_means = _get_1d_array(
        results,
        "survival_means",
    )

    if sequence_lengths.shape != survival_means.shape:
        raise ValueError(
            "sequence_lengths and survival_means must have the same shape; "
            f"got {sequence_lengths.shape} and {survival_means.shape}"
        )

    if sequence_lengths.size < 3:
        raise ValueError(
            "At least three sequence lengths are required to fit "
            "the three-parameter RB model"
        )

    if np.unique(sequence_lengths).size < 3:
        raise ValueError(
            "At least three distinct sequence lengths are required"
        )

    if np.any(sequence_lengths < 0):
        raise ValueError(
            "RB sequence lengths must be nonnegative"
        )

    if not np.allclose(
        sequence_lengths,
        np.round(sequence_lengths),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "RB sequence lengths must be integers"
        )

    if np.any(survival_means < -1e-8) or np.any(
        survival_means > 1.0 + 1e-8
    ):
        raise ValueError(
            "Survival means must lie in the interval [0, 1]"
        )

    survival_means = np.clip(
        survival_means,
        0.0,
        1.0,
    )

    sigma, weighting = _extract_fit_sigma(
        results,
        number_of_lengths=len(sequence_lengths),
    )

    initial_guess = _initial_guess(
        sequence_lengths,
        survival_means,
    )

    lower_bounds = np.array(
        [-1.0, 0.0, 0.0],
        dtype=float,
    )

    upper_bounds = np.array(
        [1.0, 1.0, 1.0],
        dtype=float,
    )

    fit_warnings: list[str] = []

    try:
        fit_arguments: dict[str, Any] = {
            "p0": initial_guess,
            "bounds": (lower_bounds, upper_bounds),
            "maxfev": 20_000,
        }

        if sigma is not None:
            fit_arguments["sigma"] = sigma
            fit_arguments["absolute_sigma"] = True

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", OptimizeWarning)

            parameters, covariance = curve_fit(
                rb_model,
                sequence_lengths,
                survival_means,
                **fit_arguments,
            )

        fit_warnings = [
            str(warning.message)
            for warning in caught_warnings
        ]

        parameters = np.asarray(
            parameters,
            dtype=float,
        )

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        if not np.all(np.isfinite(parameters)):
            raise RuntimeError(
                "The optimizer returned nonfinite fit parameters"
            )

        success = True
        message = "Fit converged"

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        parameters = np.full(3, np.nan)
        covariance = np.full((3, 3), np.nan)

        success = False
        message = f"Fit failed: {error}"

    parameter_errors = _parameter_errors(covariance)

    if success:
        fit_curve = rb_model(
            sequence_lengths,
            *parameters,
        )

        residuals = survival_means - fit_curve
        rmse = float(
            np.sqrt(np.mean(residuals**2))
        )

        sum_squared_residuals = float(
            np.sum(residuals**2)
        )

        centered = (
            survival_means
            - np.mean(survival_means)
        )

        total_sum_squares = float(
            np.sum(centered**2)
        )

        if total_sum_squares > 1e-15:
            r_squared = (
                1.0
                - sum_squared_residuals
                / total_sum_squares
            )
        else:
            r_squared = float("nan")

        degrees_of_freedom = (
            len(sequence_lengths) - 3
        )

        if sigma is not None:
            standardized_residuals = (
                residuals / sigma
            )

            chi_squared = float(
                np.sum(standardized_residuals**2)
            )

            if degrees_of_freedom > 0:
                reduced_chi_squared = (
                    chi_squared
                    / degrees_of_freedom
                )
            else:
                reduced_chi_squared = float("nan")
        else:
            standardized_residuals = np.full_like(
                residuals,
                np.nan,
            )

            chi_squared = float("nan")
            reduced_chi_squared = float("nan")

    else:
        fit_curve = np.full_like(
            survival_means,
            np.nan,
        )

        residuals = np.full_like(
            survival_means,
            np.nan,
        )

        standardized_residuals = np.full_like(
            survival_means,
            np.nan,
        )

        rmse = float("nan")
        r_squared = float("nan")
        chi_squared = float("nan")
        reduced_chi_squared = float("nan")
        degrees_of_freedom = len(sequence_lengths) - 3

    return {
        "A": float(parameters[0]),
        "A_err": float(parameter_errors[0]),
        "p": float(parameters[1]),
        "p_err": float(parameter_errors[1]),
        "B": float(parameters[2]),
        "B_err": float(parameter_errors[2]),
        "covariance": covariance,
        "fit_curve": fit_curve,
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "rmse": rmse,
        "r_squared": float(r_squared),
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "success": success,
        "message": message,
        "warnings": fit_warnings,
        "weighting": weighting,
        "sigma": (
            sigma.copy()
            if sigma is not None
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Non-Markovian diagnostics
# ---------------------------------------------------------------------------

def non_markovian_diagnostics(
    results_nm: Mapping[str, Any],
    results_mk: Mapping[str, Any],
    fit_nm: Mapping[str, Any],
    fit_mk: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compare non-Markovian and Markovian-reference RB results.

    Each returned quantity highlights a different non-Markovian signature:

    residuals_nm — NM data minus its own exponential fit: large values signal
        non-exponential (non-Markovian) decay within the NM run itself.

    residuals_nm_vs_mk — NM data minus the fitted MK curve: nonzero means the
        two models disagree, i.e., QEC did not Markovianize the noise.

    variance_ratio — cross-sequence variance NM / MK per m: a ratio > 1 means
        the NM run has excess spread, which can arise from environment memory
        that makes some Clifford sequences effectively "harder" than others.

    max_revival — largest positive jump in NM mean survival as m increases: a
        revival (increasing survival) is impossible for Markovian noise.

    p_gap — p_nm − p_mk: the signed difference in fitted decay rates.

    Parameters
    ----------
    results_nm, results_mk : dict from run_logical_rb()
    fit_nm, fit_mk : dict from fit_rb_curve()

    Returns
    -------
    dict with keys
        sequence_lengths          — shared m values (array)
        nm_fit_curve              — RB model evaluated at the NM fit parameters
        mk_fit_curve              — RB model evaluated at the MK fit parameters
        residuals_nm              — NM data − NM fit curve (array)
        residuals_nm_vs_mk        — NM data − MK fit curve (array)
        max_nonexponential_residual — max |residuals_nm|
        max_nm_mk_deviation       — max |residuals_nm_vs_mk|
        variance_nm, variance_mk  — cross-sequence variance arrays
        variance_ratio            — variance_nm / variance_mk (array; NaN where MK≈0)
        variance_difference       — variance_nm − variance_mk (array)
        mean_variance_ratio       — mean of the finite variance ratios
        max_variance_ratio        — max  of the finite variance ratios
        survival_changes_nm       — first differences of sorted NM means (array)
        max_revival               — largest positive survival increase (0 if none)
        revival_from_m, revival_to_m — m-interval where the revival occurs (NaN if none)
        p_gap                     — p_nm − p_mk (NaN if either fit failed)
        p_gap_z_score             — p_gap / sqrt(p_nm_err² + p_mk_err²)
        p_nm, p_mk                — individual fitted decay parameters
    """
    sequence_lengths_nm = _get_1d_array(
        results_nm,
        "sequence_lengths",
    )

    sequence_lengths_mk = _get_1d_array(
        results_mk,
        "sequence_lengths",
    )

    if (
        sequence_lengths_nm.shape
        != sequence_lengths_mk.shape
        or not np.array_equal(
            sequence_lengths_nm,
            sequence_lengths_mk,
        )
    ):
        raise ValueError(
            "The non-Markovian and Markovian results must use "
            "identical sequence lengths in identical order"
        )

    sequence_lengths = sequence_lengths_nm

    means_nm = _get_1d_array(
        results_nm,
        "survival_means",
    )

    means_mk = _get_1d_array(
        results_mk,
        "survival_means",
    )

    stds_nm = _get_1d_array(
        results_nm,
        "survival_stds",
    )

    stds_mk = _get_1d_array(
        results_mk,
        "survival_stds",
    )

    expected_shape = sequence_lengths.shape

    for name, array in (
        ("NM survival_means", means_nm),
        ("MK survival_means", means_mk),
        ("NM survival_stds", stds_nm),
        ("MK survival_stds", stds_mk),
    ):
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {array.shape}"
            )

    if np.any(stds_nm < 0.0) or np.any(stds_mk < 0.0):
        raise ValueError(
            "Survival standard deviations must be nonnegative"
        )

    if _fit_parameters_are_valid(fit_nm):
        nm_curve = rb_model(
            sequence_lengths,
            float(fit_nm["A"]),
            float(fit_nm["p"]),
            float(fit_nm["B"]),
        )

        residuals_nm = means_nm - nm_curve
    else:
        nm_curve = np.full_like(
            means_nm,
            np.nan,
        )

        residuals_nm = np.full_like(
            means_nm,
            np.nan,
        )

    if _fit_parameters_are_valid(fit_mk):
        mk_curve = rb_model(
            sequence_lengths,
            float(fit_mk["A"]),
            float(fit_mk["p"]),
            float(fit_mk["B"]),
        )
    else:
        # The raw Markovian means remain a useful reference when its
        # exponential fit fails.
        mk_curve = means_mk.copy()

    residuals_nm_vs_mk = means_nm - mk_curve

    variances_nm = stds_nm**2
    variances_mk = stds_mk**2

    variance_ratio = np.full_like(
        variances_nm,
        np.nan,
        dtype=float,
    )

    np.divide(
        variances_nm,
        variances_mk,
        out=variance_ratio,
        where=variances_mk > 1e-20,
    )

    variance_difference = (
        variances_nm - variances_mk
    )

    # A revival is an increase in survival as sequence length grows.
    order = np.argsort(sequence_lengths)
    sorted_lengths = sequence_lengths[order]
    sorted_nm_means = means_nm[order]

    if len(sorted_lengths) >= 2:
        survival_changes = np.diff(
            sorted_nm_means
        )

        finite_changes = np.isfinite(
            survival_changes
        )

        if np.any(finite_changes):
            finite_indices = np.flatnonzero(
                finite_changes
            )

            local_index = int(
                finite_indices[
                    np.argmax(
                        survival_changes[
                            finite_changes
                        ]
                    )
                ]
            )

            largest_increase = float(
                survival_changes[local_index]
            )

            max_revival = max(
                0.0,
                largest_increase,
            )

            if largest_increase > 0.0:
                revival_from_m = float(
                    sorted_lengths[local_index]
                )

                revival_to_m = float(
                    sorted_lengths[
                        local_index + 1
                    ]
                )
            else:
                revival_from_m = float("nan")
                revival_to_m = float("nan")
        else:
            max_revival = float("nan")
            revival_from_m = float("nan")
            revival_to_m = float("nan")
    else:
        survival_changes = np.array(
            [],
            dtype=float,
        )

        max_revival = float("nan")
        revival_from_m = float("nan")
        revival_to_m = float("nan")

    max_nonexponential_residual = (
        _maximum_absolute(residuals_nm)
    )

    max_nm_mk_deviation = (
        _maximum_absolute(
            residuals_nm_vs_mk
        )
    )

    try:
        p_nm = float(fit_nm["p"])
    except (KeyError, TypeError, ValueError):
        p_nm = float("nan")

    try:
        p_mk = float(fit_mk["p"])
    except (KeyError, TypeError, ValueError):
        p_mk = float("nan")

    if np.isfinite(p_nm) and np.isfinite(p_mk):
        p_gap = p_nm - p_mk
    else:
        p_gap = float("nan")

    try:
        p_nm_error = float(fit_nm["p_err"])
        p_mk_error = float(fit_mk["p_err"])
    except (KeyError, TypeError, ValueError):
        p_nm_error = float("nan")
        p_mk_error = float("nan")

    combined_p_error = np.sqrt(
        p_nm_error**2
        + p_mk_error**2
    )

    if (
        np.isfinite(p_gap)
        and np.isfinite(combined_p_error)
        and combined_p_error > 0.0
    ):
        p_gap_z_score = (
            p_gap / combined_p_error
        )
    else:
        p_gap_z_score = float("nan")

    finite_variance_ratios = variance_ratio[
        np.isfinite(variance_ratio)
    ]

    if finite_variance_ratios.size > 0:
        mean_variance_ratio = float(
            np.mean(finite_variance_ratios)
        )

        max_variance_ratio = float(
            np.max(finite_variance_ratios)
        )
    else:
        mean_variance_ratio = float("nan")
        max_variance_ratio = float("nan")

    return {
        "sequence_lengths": sequence_lengths.copy(),
        "nm_fit_curve": nm_curve,
        "mk_fit_curve": mk_curve,
        "residuals_nm": residuals_nm,
        "residuals_nm_vs_mk": residuals_nm_vs_mk,
        "max_nonexponential_residual": (
            max_nonexponential_residual
        ),
        "max_nm_mk_deviation": (
            max_nm_mk_deviation
        ),
        "variance_nm": variances_nm,
        "variance_mk": variances_mk,
        "variance_ratio": variance_ratio,
        "variance_difference": variance_difference,
        "mean_variance_ratio": mean_variance_ratio,
        "max_variance_ratio": max_variance_ratio,
        "survival_changes_nm": survival_changes,
        "max_revival": max_revival,
        "revival_from_m": revival_from_m,
        "revival_to_m": revival_to_m,
        "p_gap": float(p_gap),
        "p_gap_z_score": float(p_gap_z_score),
        "p_nm": p_nm,
        "p_mk": p_mk,
    }