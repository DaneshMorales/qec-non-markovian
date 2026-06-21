"""
Exact LRB simulation engine — density-matrix evolution, no shot noise.

Each Clifford sequence returns one exact survival probability

    P(m, seq) = Tr[(|0_L⟩⟨0_L| ⊗ I_E) ρ_final],

computed by evolving a density matrix through the full sequence.

The QEC step is applied as an exact Kraus channel

    Λ_QEC(ρ) = Σ_s K_s ρ K_s†,   K_s = R(s) Π_s

where the sum runs over all 2^n_stabs syndrome bit-patterns simultaneously.
This marginalises over every possible measurement outcome in one deterministic
step — no Born sampling of syndrome bits is needed.

When AncillaBitFlipNoise is used, the QEC Kraus operators are extended to
sum over all (syndrome, readout-error) pairs, folding the ancilla bit-flip
distribution into the channel itself.

Cycle ordering for one logical gate (varies by noise type):

    UnitarySENoise / MarkovianKraus / TimeVaryingKraus
        1. Logical gate        ρ ← (G ⊗ I_E) ρ (G ⊗ I_E)†
        2. Noise channel       ρ ← U_SE ρ U_SE†  |  Σ_k K_k ρ K_k†
        3. Exact QEC           ρ ← Σ_s K_s ρ K_s†
        4. Markovian reset     ρ ← Tr_E[ρ] ⊗ |0⟩⟨0|_E  (UnitarySENoise, reset_E=True only)

    PairwiseCorrelatedNoise / StreakCorrelatedNoise
        Monte Carlo over n_mc sampled error configurations; for each sample:
        1. Logical gate        ρ ← G ρ G†  (system space only)
        2. Conditional error   ρ ← Λ_error(ρ) if this step was sampled as an error
        3. Exact QEC           ρ ← Σ_s K_s ρ K_s†
        Survival probabilities are averaged over all n_mc samples.
        (reset_E is irrelevant: these models carry no quantum environment.)
"""

from __future__ import annotations
import itertools
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .codes import QECCode
from .cliffords import find_recovery_gate, sample_sequence
from .noise_models import (
    NoiseModel,
    UnitarySENoise,
    MarkovianKraus,
    TimeVaryingKraus,
    AncillaBitFlipNoise,
    PairwiseCorrelatedNoise,
    StreakCorrelatedNoise,
)
from .operators import env_zero_state


# ---------------------------------------------------------------------------
# QEC Kraus channel
# ---------------------------------------------------------------------------

def _build_qec_kraus(
    stabilizers: list[np.ndarray],
    recovery_fn,
    dim_s: int,
) -> list[np.ndarray]:
    """
    Precompute system-space Kraus operators for the exact QEC channel.

    For each syndrome bit-pattern s ∈ {0,1}^n_stabs:
        Π_s = ∏_k  (I_S + (-1)^{s_k} g_k) / 2
        K_s = R(s) Π_s

    Summing Σ_s K_s ρ K_s† over all patterns gives the exact post-QEC state
    without sampling any measurement outcome.
    """
    I = np.eye(dim_s, dtype=complex)
    ops: list[np.ndarray] = []
    for bits in itertools.product([0, 1], repeat=len(stabilizers)):
        Pi = I.copy()
        for bit, g in zip(bits, stabilizers):
            Pi = Pi @ ((I + (1.0 if bit == 0 else -1.0) * g) / 2.0)
        R = np.asarray(recovery_fn(tuple(bits)), dtype=complex)
        ops.append(R @ Pi)
    return ops


def _build_qec_kraus_ancilla_bf(
    stabilizers: list[np.ndarray],
    recovery_fn,
    dim_s: int,
    p_anc: float,
) -> list[np.ndarray]:
    """
    QEC Kraus operators with per-ancilla bit-flip readout errors.

    For each true syndrome s and each ancilla error pattern e the operator is

        K_{s,e} = sqrt(p(e)) R(s ⊕ e) Π_s,

    where p(e) = Π_k  p_anc^{e_k} (1-p_anc)^{1-e_k}.

    Summing Σ_{s,e} K_{s,e} ρ K_{s,e}† gives the correct post-QEC state
    accounting for possible wrong recoveries caused by corrupted syndromes.
    """
    n_stabs = len(stabilizers)
    I = np.eye(dim_s, dtype=complex)
    ops: list[np.ndarray] = []

    for bits in itertools.product([0, 1], repeat=n_stabs):
        Pi = I.copy()
        for bit, g in zip(bits, stabilizers):
            Pi = Pi @ ((I + (1.0 if bit == 0 else -1.0) * g) / 2.0)

        for errs in itertools.product([0, 1], repeat=n_stabs):
            p_e = np.prod(
                [p_anc if e else (1.0 - p_anc) for e in errs]
            )
            corrupted = tuple(b ^ e for b, e in zip(bits, errs))
            R = np.asarray(recovery_fn(corrupted), dtype=complex)
            ops.append(np.sqrt(p_e) * (R @ Pi))

    return ops


def _apply_channel(rho: np.ndarray, kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    """Apply ρ → Σ_k K_k ρ K_k†."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


# ---------------------------------------------------------------------------
# Markovian environment reset
# ---------------------------------------------------------------------------

def _reset_env(rho: np.ndarray, dim_s: int, dim_e: int) -> np.ndarray:
    """
    Exact Markovian reset: ρ_SE → Tr_E[ρ_SE] ⊗ |0⟩⟨0|_E.

    Traces out the environment (exact partial trace, no Born sampling) and
    reinitialises it to the vacuum.  This correctly averages over all
    environment states in one deterministic step.
    """
    rho_S = np.einsum('ikjk->ij', rho.reshape(dim_s, dim_e, dim_s, dim_e))
    env_proj = np.zeros((dim_e, dim_e), dtype=complex)
    env_proj[0, 0] = 1.0
    return np.kron(rho_S, env_proj)


# ---------------------------------------------------------------------------
# Core sequence runner
# ---------------------------------------------------------------------------

def _run_sequence(
    gate_list: Sequence[np.ndarray],
    code: QECCode,
    noise: NoiseModel,
    reset_E: bool,
    qec_kraus_sys: list[np.ndarray],
) -> float:
    """
    Compute the exact survival probability for one LRB gate sequence.

    gate_list must contain m+1 two-by-two Clifford matrices (m random gates
    followed by the recovery gate).  The function lifts each gate to the
    physical code space, applies noise, and applies the exact QEC channel.

    Returns Tr[(|0_L⟩⟨0_L| ⊗ I_E) ρ_final] clipped to [0, 1].
    """
    dim_s = 2**code.n
    zero  = np.asarray(code.encode_zero(), dtype=complex)
    zero  = zero / np.linalg.norm(zero)

    if isinstance(noise, UnitarySENoise):
        n_E   = noise.n_E
        dim_e = 2**n_E
        I_e   = np.eye(dim_e, dtype=complex)

        # Embed system-space QEC Kraus ops into S⊗E: K → K ⊗ I_E
        qec_se = [np.kron(K, I_e) for K in qec_kraus_sys]

        psi0 = np.kron(zero, env_zero_state(n_E))
        rho  = np.outer(psi0, psi0.conj())
        proj = np.kron(np.asarray(code.logical_zero_projector, dtype=complex), I_e)

        for gate in gate_list:
            G_se = np.kron(np.asarray(code.logical_unitary(gate), dtype=complex), I_e)
            rho  = G_se @ rho @ G_se.conj().T              # 1. logical gate
            rho  = noise.U_SE @ rho @ noise.U_SE.conj().T  # 2. SE noise
            rho  = _apply_channel(rho, qec_se)             # 3. exact QEC
            if reset_E:
                rho = _reset_env(rho, dim_s, dim_e)        # 4. Markovian reset

    elif isinstance(noise, (PairwiseCorrelatedNoise, StreakCorrelatedNoise)):
        # Temporally correlated noise — Monte Carlo over classical error patterns.
        # The correlations live in the distribution over which gate cycles receive
        # an error, not in a quantum environment.  For each MC sample, a joint
        # error configuration is drawn from the pairwise/streaky distribution;
        # the density matrix evolves exactly under that configuration.
        # Survival probabilities are averaged over n_mc samples.
        # reset_E is not applicable here (n_E == 0, no quantum environment).
        m    = len(gate_list) - 1
        proj = np.asarray(code.logical_zero_projector, dtype=complex)
        rng_mc = np.random.default_rng()

        total = 0.0
        for _ in range(noise.n_mc):
            step_errors = noise._sample_step_errors(m, rng_mc)
            rho = np.outer(zero, zero.conj())

            for step, gate in enumerate(gate_list):
                G   = np.asarray(code.logical_unitary(gate), dtype=complex)
                rho = G @ rho @ G.conj().T                 # 1. logical gate
                if step_errors[step]:
                    rho = _apply_channel(rho, noise.error_kraus)  # 2. corr. noise
                rho = _apply_channel(rho, qec_kraus_sys)   # 3. exact QEC

            total += float(np.clip(np.einsum('ij,ji->', proj, rho).real, 0.0, 1.0))

        return total / noise.n_mc

    else:
        # MarkovianKraus or TimeVaryingKraus — state lives in system space only
        rho  = np.outer(zero, zero.conj())
        proj = np.asarray(code.logical_zero_projector, dtype=complex)

        for step, gate in enumerate(gate_list):
            G   = np.asarray(code.logical_unitary(gate), dtype=complex)
            rho = G @ rho @ G.conj().T                     # 1. logical gate

            ops = (noise.kraus_ops if isinstance(noise, MarkovianKraus)
                   else noise.kraus_ops_list[step % len(noise.kraus_ops_list)])
            rho = _apply_channel(rho, ops)                 # 2. noise channel
            rho = _apply_channel(rho, qec_kraus_sys)       # 3. exact QEC

    return float(np.clip(np.einsum('ij,ji->', proj, rho).real, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rb_sequence_survival(
    m: int,
    code: QECCode,
    noise: NoiseModel,
    reset_E: bool = False,
    gates: Optional[Sequence[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute the exact survival probability for one LRB sequence of length m.

    No Monte Carlo: the returned value is the true expectation
    Tr[(|0_L⟩⟨0_L| ⊗ I_E) ρ_final] for this specific Clifford sequence.

    Parameters
    ----------
    m : int
        Number of random Clifford gates.  The recovery Clifford is appended
        automatically when ``gates`` is None.
    code : QECCode
    noise : NoiseModel
        Any noise model: UnitarySENoise, MarkovianKraus, TimeVaryingKraus,
        AncillaBitFlipNoise (which wraps one of the above and adds ancilla
        readout errors), PairwiseCorrelatedNoise / StreakCorrelatedNoise (and
        their polynomial/exponential convenience subclasses).
    reset_E : bool
        Apply exact partial-trace Markovian env reset after each QEC cycle.
        Only relevant for UnitarySENoise; ignored for all other noise types.
    gates : list of 2×2 arrays, optional
        Pre-built sequence [C_1, …, C_m, C_recovery].  Must have m+1 elements.
    rng : numpy.random.Generator, optional
        Used to sample the Clifford sequence when ``gates`` is None.

    Returns
    -------
    float in [0, 1]
    """
    if gates is None:
        if rng is None:
            rng = np.random.default_rng()
        seq       = sample_sequence(int(m), rng)
        gate_list = seq + [find_recovery_gate(seq)]
    else:
        gate_list = list(gates)
        if len(gate_list) != m + 1:
            raise ValueError(
                f"gates must have {m + 1} elements (m gates + recovery), "
                f"got {len(gate_list)}"
            )

    dim_s = 2**code.n
    if isinstance(noise, AncillaBitFlipNoise):
        qec_kraus   = _build_qec_kraus_ancilla_bf(
            list(code.stabilizers), code.recovery, dim_s, noise.p_anc
        )
        base_noise  = noise.base
    else:
        qec_kraus   = _build_qec_kraus(list(code.stabilizers), code.recovery, dim_s)
        base_noise  = noise

    return _run_sequence(gate_list, code, base_noise, reset_E, qec_kraus)


def run_logical_rb(
    sequence_lengths: Sequence[int],
    num_sequences: int,
    code: QECCode,
    noise: NoiseModel,
    reset_E: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """
    Run an LRB experiment: exact survival probabilities for many sequences.

    For every sequence length and every randomly drawn Clifford sequence the
    exact survival probability is computed (no Monte Carlo, no shots).  The
    only randomness is the Clifford sequence sampled for each run.

    RB decay curve: average P(m, seq) over ``num_sequences`` sequences and
    fit the result to A p^m + B.

    Parameters
    ----------
    sequence_lengths : sequence of int
        Gate counts m to probe.
    num_sequences : int
        Independent Clifford sequences per length.
    code : QECCode
    noise : NoiseModel
        Any noise model: UnitarySENoise, MarkovianKraus, TimeVaryingKraus,
        AncillaBitFlipNoise, PairwiseCorrelatedNoise / StreakCorrelatedNoise
        (and their polynomial/exponential subclasses).
    reset_E : bool
        Apply exact partial-trace Markovian env reset after every QEC cycle.
        Only relevant for UnitarySENoise; ignored for all other noise types.
    seed : int, optional
        Seed for the Clifford-sequence RNG.

    Returns
    -------
    dict with keys
        sequence_lengths : int array
        survival_means   : mean exact survival probability per length
        survival_stds    : std across sequences
        survival_sems    : SEM across sequences
        all_survivals    : shape (num_lengths, num_sequences) exact values
        all_seq_means    : alias for all_survivals (backward compat)
        reset_E          : bool
        code_name        : str
        noise_type       : str
    """
    lengths = np.asarray(list(sequence_lengths), dtype=int)
    dim_s   = 2**code.n
    rng     = np.random.default_rng(seed)

    if isinstance(noise, AncillaBitFlipNoise):
        qec_kraus  = _build_qec_kraus_ancilla_bf(
            list(code.stabilizers), code.recovery, dim_s, noise.p_anc
        )
        base_noise = noise.base
    else:
        qec_kraus  = _build_qec_kraus(list(code.stabilizers), code.recovery, dim_s)
        base_noise = noise

    all_survivals = np.empty((len(lengths), int(num_sequences)), dtype=float)

    for li, m in enumerate(lengths):
        for si in range(num_sequences):
            seq       = sample_sequence(int(m), rng)
            gate_list = seq + [find_recovery_gate(seq)]
            all_survivals[li, si] = _run_sequence(gate_list, code, base_noise, reset_E, qec_kraus)

    survival_means = np.mean(all_survivals, axis=1)
    survival_stds  = np.std(all_survivals,  axis=1, ddof=0)
    survival_sems  = survival_stds / np.sqrt(num_sequences)

    return {
        "sequence_lengths": lengths,
        "survival_means":   survival_means,
        "survival_stds":    survival_stds,
        "survival_sems":    survival_sems,
        "all_survivals":    all_survivals,
        "all_seq_means":    all_survivals,   # alias for backward compat
        "reset_E":          reset_E,
        "code_name":        type(code).__name__,
        "noise_type":       type(noise).__name__,
    }
