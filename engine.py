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

Performance (UnitarySENoise fast path in run_logical_rb)
---------------------------------------------------------
For UnitarySENoise (and AncillaBitFlipNoise wrapping it), run_logical_rb
precomputes the following once per experiment rather than once per sequence:

  • All 24 combined gate+noise SE unitaries:
        N_i = U_SE @ kron(logical_unitary(CLIFFORDS[i]), I_E)
    so each gate step reduces to one matrix multiply instead of two.

  • Pauli-structure QEC fast path (CSS codes such as Steane and RepetitionCode):
    When every code.recovery(s) is a Pauli (monomial) matrix the QEC channel
    collapses to

        Λ(ρ_SE) = W_SE · B_SE · W_SE†,    B_SE = W_SE† · ρ_depol_SE · W_SE

    where W_SE = kron(encoder, I_E) is 256×4 and B_SE is a 4×4 matrix.
    ρ_depol_SE is computed from 64 signed row/column permutations of ρ_SE
    (O(256²) each, no matrix multiplications).  This replaces 64 dense
    256×256 Kraus applies (≈7 s per gate for Steane) with ≈140 ms.

  • SE-embedded QEC Kraus ops (fallback when recovery is not a Pauli).

Setting n_jobs > 1 enables thread-level parallelism via ThreadPoolExecutor.
NumPy releases the GIL during BLAS matrix operations, so threads run in
parallel for the dominant computation.  Use n_jobs=-1 for all CPU cores.
"""

from __future__ import annotations
import itertools
from collections.abc import Sequence
from typing import Optional

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable=None, *, total=None, desc=None, unit=None, disable=False, **kw):  # type: ignore[misc]
        return iterable if iterable is not None else (lambda x: x)

from .codes import QECCode
from .cliffords import (
    CLIFFORDS,
    _matrix_key,
    _KEY_TO_IDX,
    find_recovery_gate,
    sample_sequence,
)
from .noise_models import (
    NoiseModel,
    UnitarySENoise,
    MarkovianKraus,
    TimeVaryingKraus,
    AncillaBitFlipNoise,
    PairwiseCorrelatedNoise,
    StreakCorrelatedNoise,
    PartialDepolarizingKraus,
)
from .operators import I2, env_zero_state


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

    Uses dynamic programming to share partial projectors across syndrome
    patterns: for n_stabs stabilizers this requires 2^(n_stabs+1) − 2 matrix
    multiplications instead of n_stabs × 2^n_stabs (3× fewer for Steane).
    """
    I = np.eye(dim_s, dtype=complex)
    # DP table: bits_prefix -> partial projector Π_{s_0,...,s_{k-1}}
    partial: dict[tuple[int, ...], np.ndarray] = {(): I}
    for g in stabilizers:
        P_plus  = (I + g) / 2.0
        P_minus = (I - g) / 2.0
        next_partial: dict[tuple[int, ...], np.ndarray] = {}
        for bits, Pi in partial.items():
            next_partial[bits + (0,)] = Pi @ P_plus
            next_partial[bits + (1,)] = Pi @ P_minus
        partial = next_partial

    ops: list[np.ndarray] = []
    for bits, Pi in partial.items():
        R = np.asarray(recovery_fn(bits), dtype=complex)
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
    out = np.zeros_like(rho)
    for K in kraus_ops:
        out += K @ rho @ K.conj().T
    return out


def _apply_channel_adj(
    rho: np.ndarray,
    kraus_fwd: Sequence[np.ndarray],
    kraus_adj: Sequence[np.ndarray],
) -> np.ndarray:
    """Apply ρ → Σ_k K_k ρ K_k† using precomputed adjoints (avoids .conj().T per call)."""
    out = np.zeros_like(rho)
    for K, Kd in zip(kraus_fwd, kraus_adj):
        out += K @ rho @ Kd
    return out


# ---------------------------------------------------------------------------
# Pauli-structure QEC fast path
# ---------------------------------------------------------------------------

def _try_build_pauli_qec_cache(
    code: QECCode,
    n_E: int,
) -> Optional[dict]:
    """
    Try to build the Pauli-structure QEC cache.

    If every code.recovery(s) is a monomial matrix (i.e. a Pauli — one nonzero
    entry per row with magnitude 1), the QEC channel simplifies to

        Λ(ρ_SE) = W_SE · B_SE · W_SE†

    where W_SE = kron(encoder, I_E)  (256×4 for Steane + 1 env qubit)
    and   B_SE = W_SE† · ρ_depol_SE · W_SE  (4×4 matrix).

    ρ_depol_SE = Σ_s  phase_outer_s * ρ_SE[perm_s, :][:, perm_s]
    is computed by signed row/col permutations (O(dim_SE²) per syndrome),
    with no matrix multiplications.

    Returns a dict with precomputed data, or None if any recovery is not a
    Pauli (in which case the standard dense Kraus path is used instead).
    """
    dim_s = 2**code.n
    dim_e = 2**n_E
    dim_se = dim_s * dim_e
    I_e = np.eye(dim_e, dtype=complex)
    W = np.asarray(code.encoder(), dtype=complex)   # dim_s × 2
    W_SE = np.kron(W, I_e)                          # dim_se × (2 * dim_e)

    n_stabs = len(code.stabilizers)
    perm_se_list:    list[np.ndarray] = []
    phase_outer_list: list[np.ndarray] = []

    sys_indices = np.arange(dim_s)
    env_indices = np.arange(dim_e)

    for bits in itertools.product([0, 1], repeat=n_stabs):
        R_s = np.asarray(code.recovery(bits), dtype=complex)

        # Detect monomial structure: exactly one nonzero per row
        nz_counts = np.count_nonzero(np.abs(R_s) > 1e-10, axis=1)
        if not np.all(nz_counts == 1):
            return None   # Not a Pauli — fall back to dense Kraus

        perm_s  = np.argmax(np.abs(R_s), axis=1)        # length dim_s
        phase_s = R_s[sys_indices, perm_s]               # length dim_s

        # Expand perm/phase to SE space (system outer, env inner)
        # SE index for (sys=i, env=a) is i * dim_e + a
        perm_se  = np.empty(dim_se, dtype=int)
        phase_se = np.empty(dim_se, dtype=complex)
        for a in env_indices:
            idx = sys_indices * dim_e + a
            perm_se[idx]  = perm_s * dim_e + a
            phase_se[idx] = phase_s

        perm_se_list.append(perm_se)
        phase_outer_list.append(np.outer(phase_se, phase_se.conj()))

    return {
        "perm_se_list":     perm_se_list,
        "phase_outer_list": phase_outer_list,
        "W_SE":             W_SE,
        "W_SE_adj":         W_SE.conj().T.copy(),
    }


def _apply_channel_batch(rho_batch: np.ndarray, ops: list[np.ndarray]) -> np.ndarray:
    """
    Apply a Kraus channel to a batch of density matrices.

    rho_batch : (n_mc, d, d)
    ops       : list of (d, d) Kraus operators

    numpy matmul broadcasts (d,d) @ (n,d,d) and (n,d,d) @ (d,d) correctly,
    so this is a vectorised loop over n_mc without any Python iteration.
    """
    out = np.zeros_like(rho_batch)
    for K in ops:
        out += K @ rho_batch @ K.conj().T
    return out


def _apply_qec_pauli_fast(rho_se: np.ndarray, cache: dict) -> np.ndarray:
    """
    Apply the QEC channel using the Pauli-structure cache.

    Computes  ρ_depol_SE = Σ_s phase_outer_s * ρ_SE[perm_s, perm_s]
    then returns  W_SE · (W_SE† · ρ_depol_SE · W_SE) · W_SE†.
    """
    rho_depol = np.zeros_like(rho_se)
    for perm, ph_outer in zip(cache["perm_se_list"], cache["phase_outer_list"]):
        rho_depol += ph_outer * rho_se[np.ix_(perm, perm)]
    W_SE     = cache["W_SE"]
    W_SE_adj = cache["W_SE_adj"]
    B_SE = W_SE_adj @ rho_depol @ W_SE
    return W_SE @ B_SE @ W_SE_adj


# ---------------------------------------------------------------------------
# Precomputation helpers for UnitarySENoise fast path
# ---------------------------------------------------------------------------

def _precompute_se_cache(
    code: QECCode,
    noise: UnitarySENoise,
    qec_kraus_sys: list[np.ndarray],
) -> tuple[list, list, list, list, np.ndarray, np.ndarray]:
    """
    Precompute all SE-space matrices needed for a UnitarySENoise LRB run.

    For each of the 24 single-qubit Cliffords C_i the combined gate+noise
    unitary in S⊗E space is

        N_i = U_SE @ kron(logical_unitary(C_i), I_E).

    Applying gate i followed by noise is then a single matrix multiply

        rho ← N_i @ rho @ N_i†

    instead of two.  The QEC Kraus ops and their adjoints are also embedded
    into S⊗E space once here (not once per sequence).

    Returns
    -------
    noisy_fwd  : list of 24 arrays, noisy_fwd[i] = N_i
    noisy_adj  : list of 24 arrays, noisy_adj[i] = N_i†
    qec_fwd    : SE-embedded QEC Kraus ops
    qec_adj    : adjoints of qec_fwd
    psi0       : initial state vector |0_L⟩ ⊗ |0_E⟩
    proj       : survival projector |0_L⟩⟨0_L| ⊗ I_E in SE space
    """
    n_E   = noise.n_E
    dim_e = 2**n_E
    I_e   = np.eye(dim_e, dtype=complex)

    noisy_fwd: list[np.ndarray] = []
    noisy_adj: list[np.ndarray] = []
    for C in CLIFFORDS:
        G_code = np.asarray(code.logical_unitary(C), dtype=complex)
        N      = noise.U_SE @ np.kron(G_code, I_e)
        noisy_fwd.append(N)
        noisy_adj.append(N.conj().T)

    qec_fwd = [np.kron(K, I_e) for K in qec_kraus_sys]
    qec_adj = [K.conj().T for K in qec_fwd]

    zero = np.asarray(code.encode_zero(), dtype=complex)
    zero = zero / np.linalg.norm(zero)
    psi0 = np.kron(zero, env_zero_state(n_E))
    proj = np.kron(np.asarray(code.logical_zero_projector, dtype=complex), I_e)

    return noisy_fwd, noisy_adj, qec_fwd, qec_adj, psi0, proj


def _recovery_idx(index_seq: list[int]) -> int:
    """Return the Clifford index of the recovery gate for a sequence of indices."""
    U = I2.copy()
    for idx in index_seq:
        U = CLIFFORDS[idx] @ U
    return _KEY_TO_IDX[_matrix_key(U.conj().T)]


# ---------------------------------------------------------------------------
# Thread-based parallel task for UnitarySENoise
# ---------------------------------------------------------------------------

def _se_thread_task(args: tuple) -> float:
    """
    Evaluate one LRB sequence survival probability in a thread worker.

    Uses ThreadPoolExecutor (not ProcessPoolExecutor) so that the precomputed
    SE-space matrices are shared in memory across threads with no copying.
    NumPy releases the GIL for matrix operations, so threads run in parallel
    for the dominant (BLAS-level) computation.

    args
    ----
    gate_indices  : list[int]       Clifford indices (m random gates + recovery)
    fwd           : list            noisy_fwd cache from _precompute_se_cache
    adj           : list            noisy_adj cache
    qec_fwd       : list            SE-embedded QEC Kraus ops (fallback only)
    qec_adj       : list            adjoints of qec_fwd (fallback only)
    pauli_cache   : dict | None     Pauli QEC cache; if not None, takes priority
    dim_s, dim_e  : int             Hilbert space dimensions
    psi0          : ndarray         initial state |0_L⟩ ⊗ |0_E⟩
    proj          : ndarray         survival projector
    reset_E       : bool
    """
    gate_indices, fwd, adj, qec_fwd, qec_adj, pauli_cache, dim_s, dim_e, psi0, proj, reset_E = args
    rho = np.outer(psi0, psi0.conj())
    for idx in gate_indices:
        rho = fwd[idx] @ rho @ adj[idx]
        if pauli_cache is not None:
            rho = _apply_qec_pauli_fast(rho, pauli_cache)
        else:
            rho = _apply_channel_adj(rho, qec_fwd, qec_adj)
        if reset_E:
            rho = _reset_env(rho, dim_s, dim_e)
    return float(np.clip(np.einsum("ij,ji->", proj, rho).real, 0.0, 1.0))


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
        # Batched MC path for all correlated noise types.
        #
        # All n_mc error patterns are sampled at once via a matrix multiply in
        # _sample_step_errors_batch, and all n_mc density matrices evolve as a
        # single (n_mc, d, d) batch — numpy broadcasts (d,d)@(n,d,d) natively.
        #
        # apply_error_batch dispatches to either the O(d²) convex-mixture for
        # DepolarizingPairwiseNoise / DepolarizingStreakNoise or the general
        # Kraus loop (vectorised over the batch) for arbitrary error channels.
        m      = len(gate_list) - 1
        n_mc   = noise.n_mc
        proj   = np.asarray(code.logical_zero_projector, dtype=complex)
        rng_mc = np.random.default_rng()

        # (n_mc, m+1) int8: 1 = error fires at that (sample, step)
        step_errors_batch = noise._sample_step_errors_batch(m, n_mc, rng_mc)

        rho0      = np.outer(zero, zero.conj())
        rho_batch = np.tile(rho0, (n_mc, 1, 1))            # (n_mc, d, d)

        for step, gate in enumerate(gate_list):
            G    = np.asarray(code.logical_unitary(gate), dtype=complex)
            Gadj = G.conj().T
            rho_batch = G @ rho_batch @ Gadj               # 1. gate (broadcast)

            fired = step_errors_batch[:, step].astype(bool)
            if fired.any():                                # 2. corr. noise
                rho_batch[fired] = noise.apply_error_batch(rho_batch[fired])

            rho_batch = _apply_channel_batch(rho_batch, qec_kraus_sys)  # 3. QEC

        # Tr[proj @ rho_n] = Σ_{ij} proj_{ij} rho_n_{ji}
        survivals = np.einsum("ij,nji->n", proj, rho_batch).real
        return float(np.mean(np.clip(survivals, 0.0, 1.0)))

    elif isinstance(noise, PartialDepolarizingKraus):
        # Marginalized independent baseline from gen_markovian_baseline().
        # Λ_t(ρ) = (1-p_t)ρ + p_t · I/dim_s  — applied as a direct convex
        # mixture (O(dim_s²)), avoiding any Kraus operator loop.
        rho  = np.outer(zero, zero.conj())
        proj = np.asarray(code.logical_zero_projector, dtype=complex)

        for step, gate in enumerate(gate_list):
            G   = np.asarray(code.logical_unitary(gate), dtype=complex)
            rho = G @ rho @ G.conj().T                          # 1. logical gate
            p_t = noise.p_list[step % len(noise.p_list)]
            rho = (1.0 - p_t) * rho + p_t * noise._I_d         # 2. partial depol
            rho = _apply_channel(rho, qec_kraus_sys)            # 3. exact QEC

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
    apply_qec: bool = True,
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
    apply_qec : bool
        Apply syndrome extraction and recovery after every gate (default True).
        Set to False to run bare LRB with no error correction — useful for
        comparing raw SE-noise decay against the QEC-protected decay to probe
        whether QEC Markovianizes the noise.
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
    if not apply_qec:
        # Identity channel: no syndrome extraction, no recovery.
        qec_kraus  = [np.eye(dim_s, dtype=complex)]
        base_noise = noise.base if isinstance(noise, AncillaBitFlipNoise) else noise
    elif isinstance(noise, AncillaBitFlipNoise):
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
    apply_qec: bool = True,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    show_progress: bool = True,
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
    apply_qec : bool
        Apply syndrome extraction and recovery after every gate (default True).
        Set to False to run bare LRB with no error correction — useful for
        comparing the raw SE-noise decay against the QEC-protected decay to
        probe whether QEC Markovianizes the noise.
    seed : int, optional
        Seed for the Clifford-sequence RNG.
    n_jobs : int
        Number of parallel worker processes for the sequence loop.
        ``1`` (default) runs sequentially.  ``-1`` uses all available CPUs.
        Only effective for UnitarySENoise; other noise types always run
        sequentially (their inner loops are already fast or involve
        stochastic state that is hard to parallelise).
    show_progress : bool
        Show a tqdm progress bar over sequences (default True).
        Requires the ``tqdm`` package; silently ignored if not installed.

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
        apply_qec        : bool
        code_name        : str
        noise_type       : str
    """
    lengths = np.asarray(list(sequence_lengths), dtype=int)
    dim_s   = 2**code.n
    rng     = np.random.default_rng(seed)

    # Resolve base noise (strip AncillaBitFlipNoise wrapper).
    if isinstance(noise, AncillaBitFlipNoise):
        base_noise = noise.base
    else:
        base_noise = noise

    all_survivals = np.empty((len(lengths), int(num_sequences)), dtype=float)

    if isinstance(base_noise, UnitarySENoise):
        # ----------------------------------------------------------------
        # Fast path for UnitarySENoise
        # ----------------------------------------------------------------
        # For CSS codes (Steane, RepetitionCode) all recovery operators are
        # Paulis.  The QEC channel then collapses to two cheap 256×4 matmuls
        # instead of 64 dense 256×256 Kraus applies.  We try this first so
        # that _build_qec_kraus (expensive for large codes) is never called
        # when the Pauli path succeeds.
        # ----------------------------------------------------------------
        if apply_qec and not isinstance(noise, AncillaBitFlipNoise):
            pauli_cache = _try_build_pauli_qec_cache(code, base_noise.n_E)
        else:
            pauli_cache = None

        if pauli_cache is not None:
            # Pauli fast path: dense Kraus ops not needed.
            qec_kraus = [np.eye(dim_s, dtype=complex)]   # placeholder, unused
        elif not apply_qec:
            qec_kraus = [np.eye(dim_s, dtype=complex)]
        elif isinstance(noise, AncillaBitFlipNoise):
            qec_kraus = _build_qec_kraus_ancilla_bf(
                list(code.stabilizers), code.recovery, dim_s, noise.p_anc
            )
        else:
            qec_kraus = _build_qec_kraus(
                list(code.stabilizers), code.recovery, dim_s
            )

        noisy_fwd, noisy_adj, qec_fwd, qec_adj, psi0, proj = _precompute_se_cache(
            code, base_noise, qec_kraus
        )
        dim_e = 2**base_noise.n_E

        # Generate all gate-index sequences deterministically from seed
        all_tasks: list[tuple[int, int, list[int]]] = []
        for li, m in enumerate(lengths):
            for si in range(num_sequences):
                idx_seq  = list(rng.integers(0, 24, size=int(m)))
                rec_idx  = _recovery_idx(idx_seq)
                all_tasks.append((li, si, idx_seq + [rec_idx]))

        bar_kw = dict(
            total=len(all_tasks),
            desc=f"LRB ({type(code).__name__})",
            unit="seq",
            disable=not show_progress,
        )
        if n_jobs == 1:
            for li, si, gate_indices in _tqdm(all_tasks, **bar_kw):
                rho = np.outer(psi0, psi0.conj())
                for idx in gate_indices:
                    rho = noisy_fwd[idx] @ rho @ noisy_adj[idx]
                    if pauli_cache is not None:
                        rho = _apply_qec_pauli_fast(rho, pauli_cache)
                    else:
                        rho = _apply_channel_adj(rho, qec_fwd, qec_adj)
                    if reset_E:
                        rho = _reset_env(rho, dim_s, dim_e)
                all_survivals[li, si] = float(
                    np.clip(np.einsum("ij,ji->", proj, rho).real, 0.0, 1.0)
                )
        else:
            import concurrent.futures
            max_workers = None if n_jobs == -1 else int(n_jobs)
            # Threads share the precomputed arrays in memory (no copy overhead).
            # NumPy releases the GIL during BLAS-level matrix ops, so threads
            # run in parallel for the dominant computation.
            task_args = [
                (gi, noisy_fwd, noisy_adj, qec_fwd, qec_adj, pauli_cache,
                 dim_s, dim_e, psi0, proj, reset_E)
                for (_, _, gi) in all_tasks
            ]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as pool:
                results = list(_tqdm(pool.map(_se_thread_task, task_args), **bar_kw))
            for (li, si, _), val in zip(all_tasks, results):
                all_survivals[li, si] = val

    else:
        # Fallback path for correlated / Markovian-Kraus noise types.
        # Build QEC Kraus ops here (deferred to avoid paying the cost when
        # the UnitarySENoise fast path is taken instead).
        if not apply_qec:
            qec_kraus = [np.eye(dim_s, dtype=complex)]
        elif isinstance(noise, AncillaBitFlipNoise):
            qec_kraus = _build_qec_kraus_ancilla_bf(
                list(code.stabilizers), code.recovery, dim_s, noise.p_anc
            )
        else:
            qec_kraus = _build_qec_kraus(
                list(code.stabilizers), code.recovery, dim_s
            )

        # Pre-generate all gate sequences sequentially (shared rng, deterministic).
        # Each (li, si, gate_list) triple is then an independent unit of work.
        flat_tasks = [
            (li, si, sample_sequence(int(m), rng))
            for li, m in enumerate(lengths)
            for si in range(num_sequences)
        ]
        gate_lists = [seq + [find_recovery_gate(seq)] for (_, _, seq) in flat_tasks]

        bar_kw = dict(
            total=len(flat_tasks),
            desc=f"LRB ({type(code).__name__})",
            unit="seq",
            disable=not show_progress,
        )

        if n_jobs == 1:
            for (li, si, _), gate_list in _tqdm(zip(flat_tasks, gate_lists), **bar_kw):
                all_survivals[li, si] = _run_sequence(
                    gate_list, code, base_noise, reset_E, qec_kraus
                )
        else:
            import concurrent.futures
            max_workers = None if n_jobs == -1 else int(n_jobs)
            # Each _run_sequence call creates its own rng_mc internally,
            # so threads are independent with no shared mutable state.
            # NumPy releases the GIL for matrix ops — threads run in parallel.
            _run = lambda gl: _run_sequence(gl, code, base_noise, reset_E, qec_kraus)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(_tqdm(pool.map(_run, gate_lists), **bar_kw))
            for (li, si, _), val in zip(flat_tasks, results):
                all_survivals[li, si] = val

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
        "apply_qec":        apply_qec,
        "code_name":        type(code).__name__,
        "noise_type":       type(noise).__name__,
    }
