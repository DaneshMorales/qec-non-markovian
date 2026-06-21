"""
demo_logical_rb.py
==================

Demonstrate logical randomized benchmarking under persistent-environment
(non-Markovian) noise.

Step 1 — Three-qubit repetition code + XX exchange coupling
Step 2 — Steane [[7,1,3]] code + ZX memory coupling
Step 3 — Correction-frequency sweep (Steane, fixed total interaction time)

Run:  python demo_logical_rb.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from logical_rb import (
    RepetitionCode, SteaneCode,
    UnitarySENoise,
    find_recovery_gate, sample_sequence,
    fit_rb_curve, non_markovian_diagnostics,
    plot_comparison, plot_correction_frequency_sweep,
    rb_sequence_survival, run_logical_rb,
    hamiltonian_coupling, xx_coupling,
)
from logical_rb.operators import I2, X, Z, lift

OUTDIR = PROJECT_DIR


# ---------------------------------------------------------------------------
# ZX memory coupling (Steane demo)
# ---------------------------------------------------------------------------

def zx_memory_coupling(n_sys: int, n_E: int, J: float, tau: float) -> np.ndarray:
    """
    H_SE = J Σ_k Z_k^S X_0^E,   U_SE = exp(-i H_SE tau).

    |0>_E is not an X eigenstate, so this interaction genuinely entangles S
    and E and produces non-Markovian memory.
    """
    n_total = n_sys + n_E
    dim     = 2**n_total
    H_SE    = np.zeros((dim, dim), dtype=complex)
    X_E0    = lift(X, n_sys, n_total)
    for k in range(n_sys):
        H_SE += J * lift(Z, k, n_total) @ X_E0
    return hamiltonian_coupling(H_SE, tau)


# ---------------------------------------------------------------------------
# Correction-frequency simulation
# ---------------------------------------------------------------------------

def _expand_gates(gate_list: list[np.ndarray], r: int) -> list[np.ndarray]:
    """Replace each gate C with [C, I, I, ..., I] (r entries total)."""
    expanded = []
    for gate in gate_list:
        expanded.append(np.asarray(gate, dtype=complex))
        expanded.extend([I2.copy()] * (r - 1))
    return expanded


def run_with_substeps(
    sequence_lengths: Sequence[int],
    num_sequences: int,
    code: SteaneCode,
    n_E: int,
    J: float,
    tau_total: float,
    r: int,
    reset_E: bool = False,
    seed: int | None = None,
) -> dict:
    """
    LRB with r QEC rounds per logical gate.

    Total SE interaction time per logical gate stays fixed at tau_total.
    Each substep uses tau_step = tau_total / r.
    """
    tau_step = tau_total / r
    noise    = UnitarySENoise(
        zx_memory_coupling(code.n, n_E, J, tau_step),
        n_E,
    )
    lengths  = np.asarray(list(sequence_lengths), dtype=int)
    rng      = np.random.default_rng(seed)

    all_survivals = np.empty((len(lengths), num_sequences), dtype=float)

    for li, m in enumerate(lengths):
        for si in range(num_sequences):
            seq       = sample_sequence(int(m), rng)
            base      = seq + [find_recovery_gate(seq)]
            expanded  = _expand_gates(base, r)
            expanded_m = len(expanded) - 1
            all_survivals[li, si] = rb_sequence_survival(
                m=expanded_m, code=code, noise=noise,
                reset_E=reset_E, gates=expanded, rng=rng,
            )

    survival_means = np.mean(all_survivals, axis=1)
    survival_stds  = np.std(all_survivals,  axis=1, ddof=0)
    survival_sems  = survival_stds / np.sqrt(num_sequences)

    return {
        "sequence_lengths": lengths,
        "survival_means":   survival_means,
        "survival_stds":    survival_stds,
        "survival_sems":    survival_sems,
        "all_seq_means":    all_survivals,
        "all_survivals":    all_survivals,
        "reset_E":          reset_E,
        "code_name":        type(code).__name__,
        "r": r,
        "tau_total": tau_total,
        "tau_step":  tau_step,
    }


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def _print_fit(label: str, fit: dict) -> None:
    p, p_err = float(fit["p"]), float(fit["p_err"])
    if np.isfinite(p):
        suffix = f" ± {p_err:.4f}" if np.isfinite(p_err) else ""
        print(f"  {label:<26} p = {p:.4f}{suffix}")
    else:
        print(f"  {label:<26} {fit.get('message', 'fit failed')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ============================================================
    # Step 1 — Repetition code
    # ============================================================
    print("=" * 60)
    print("Step 1 — Three-qubit repetition code (XX exchange)")
    print("=" * 60)

    rep   = RepetitionCode()
    n_E_r = 1
    noise_rep = UnitarySENoise(xx_coupling(n_sys=rep.n, n_E=n_E_r, g=0.15, tau=0.5), n_E_r)

    lengths_rep = [1, 2, 3, 5, 8, 12, 18, 25]
    seed_rep    = 0

    print("  Non-Markovian...")
    res_nm_rep = run_logical_rb(lengths_rep, 80, rep, noise_rep, reset_E=False, seed=seed_rep)
    print("  Markovian reference...")
    res_mk_rep = run_logical_rb(lengths_rep, 80, rep, noise_rep, reset_E=True,  seed=seed_rep)

    fit_nm_rep = fit_rb_curve(res_nm_rep)
    fit_mk_rep = fit_rb_curve(res_mk_rep)
    diag_rep   = non_markovian_diagnostics(res_nm_rep, res_mk_rep, fit_nm_rep, fit_mk_rep)

    _print_fit("Non-Markovian:",    fit_nm_rep)
    _print_fit("Markovian ref:",    fit_mk_rep)
    print(f"  p gap (NM-MK):             {diag_rep['p_gap']:.4f}")
    print(f"  max NM-MK deviation:       {diag_rep['max_nm_mk_deviation']:.4f}")

    fig1 = plot_comparison(res_nm_rep, res_mk_rep, fit_nm_rep, fit_mk_rep, diag_rep,
                           title="Step 1 — Repetition code (XX exchange)",
                           save_path=OUTDIR / "step1_rep_code.png")
    plt.close(fig1)

    # ============================================================
    # Step 2 — Steane code
    # ============================================================
    print()
    print("=" * 60)
    print("Step 2 — Steane [[7,1,3]] code (ZX memory coupling)")
    print("=" * 60)

    steane = SteaneCode()
    n_E_s  = 1
    noise_steane = UnitarySENoise(zx_memory_coupling(steane.n, n_E_s, J=0.25, tau=0.6), n_E_s)

    lengths_steane = [1, 2, 3, 5, 8, 12, 20, 30]
    seed_steane    = 2

    print("  Non-Markovian...")
    res_nm_s = run_logical_rb(lengths_steane, 50, steane, noise_steane, reset_E=False, seed=seed_steane)
    print("  Markovian reference...")
    res_mk_s = run_logical_rb(lengths_steane, 50, steane, noise_steane, reset_E=True,  seed=seed_steane)

    fit_nm_s = fit_rb_curve(res_nm_s)
    fit_mk_s = fit_rb_curve(res_mk_s)
    diag_s   = non_markovian_diagnostics(res_nm_s, res_mk_s, fit_nm_s, fit_mk_s)

    _print_fit("Non-Markovian:",    fit_nm_s)
    _print_fit("Markovian ref:",    fit_mk_s)
    print(f"  p gap (NM-MK):             {diag_s['p_gap']:.4f}")
    print(f"  max NM-MK deviation:       {diag_s['max_nm_mk_deviation']:.4f}")

    fig2 = plot_comparison(res_nm_s, res_mk_s, fit_nm_s, fit_mk_s, diag_s,
                           title="Step 2 — Steane [[7,1,3]] (ZX memory coupling)",
                           save_path=OUTDIR / "step2_steane.png")
    plt.close(fig2)

    # ============================================================
    # Step 3 — Correction-frequency sweep
    # ============================================================
    print()
    print("=" * 60)
    print("Step 3 — Correction-frequency sweep (Steane, fixed tau_total)")
    print("=" * 60)

    sweep_lengths = lengths_steane[:6]
    sweep_results = []

    for r in [1, 2, 4, 8]:
        tau_step = 0.6 / r
        print(f"  r={r}, tau_step={tau_step:.3f}...")
        result = run_with_substeps(
            sweep_lengths, 50, steane, n_E_s, J=0.25, tau_total=0.6,
            r=r, reset_E=False, seed=10,
        )
        fit = fit_rb_curve(result)
        p   = float(fit["p"])
        result["label"] = f"r={r}, tau={tau_step:.3f}" + (f", p={p:.3f}" if np.isfinite(p) else "")
        sweep_results.append(result)
        _print_fit(f"r={r}, tau={tau_step:.3f}:", fit)

    fig3 = plot_correction_frequency_sweep(
        sweep_results, "QEC rounds per logical gate (fixed total tau)",
        save_path=OUTDIR / "step3_freq_sweep.png",
    )
    plt.close(fig3)

    print()
    print(f"Done. PNGs written to {OUTDIR}")
    print("  step1_rep_code.png  step2_steane.png  step3_freq_sweep.png")


if __name__ == "__main__":
    main()
