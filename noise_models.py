"""
Noise model classes and SE coupling unitary constructors.

Noise model classes (pass to run_logical_rb / rb_sequence_survival)
--------------------------------------------------------------------
UnitarySENoise(U_SE, n_E)
    Non-Markovian: persistent environment coupled via a fixed unitary U_SE
    applied after every logical gate.  The environment qubits are kept across
    QEC cycles, so quantum information can leak between them (non-Markovian
    memory).  Pass reset_E=True to run_logical_rb for the Markovian reference
    (exact partial-trace env reset after each QEC cycle).

MarkovianKraus(kraus_ops)
    Markovian CPTP channel; the same Kraus operators every cycle, no environment.

TimeVaryingKraus(kraus_ops_list)
    Time-structured CPTP channel without a quantum environment: the Kraus
    operators change deterministically per gate cycle (index wraps periodically).

AncillaBitFlipNoise(base, p_anc)
    Wraps any noise model and adds independent bit-flip errors on each ancilla
    (syndrome extraction) qubit.  Each ancilla readout is flipped with
    probability p_anc ∈ [0, 0.5], so the decoder sees a corrupted syndrome
    and may apply the wrong recovery operator.  Data-qubit noise is handled
    by the wrapped base model.  p_anc = 0 is exactly equivalent to base alone;
    p_anc = 0.5 gives maximally uninformative (random) syndrome readout.

PairwiseCorrelatedNoise(interaction_func, error_kraus, n_mc)
    Temporally correlated noise: each pair of gate cycles (t1, t2) can trigger
    a joint error event with probability p(t2 − t1) given by interaction_func.
    When a pair fires, error_kraus is applied at both t1 and t2 (XOR composition
    for overlapping pairs).  The engine evaluates exact survival by averaging
    n_mc independently sampled error configurations (Monte Carlo over the
    classical correlation structure, not over quantum outcomes).

StreakCorrelatedNoise(interaction_func, error_kraus, n_mc)
    Burst-style temporally correlated noise: a burst spanning [t1, t2] fires
    with probability p(t2 − t1).  Each step inside a fired burst independently
    receives an error with probability 0.5 (XOR'd across overlapping bursts).

Convenience subclasses for the common decay functions
------------------------------------------------------
PairwisePolyNoise(A, q, n, error_kraus)   pairwise, p(Δt) = A·q / Δt^n
PairwiseExpNoise(A, q, n, error_kraus)    pairwise, p(Δt) = A·q / n^Δt
StreakPolyNoise(A, q, n, error_kraus)     streaky,  p(Δt) = A·q / Δt^n
StreakExpNoise(A, q, n, error_kraus)      streaky,  p(Δt) = A·q / n^Δt

Both correlated classes also expose:
    calc_marginals_per_cycle(m)   — analytical per-cycle marginal error prob.
    gen_markovian_baseline(m)     — TimeVaryingKraus with matched marginals.

SE coupling unitary constructors
---------------------------------
hamiltonian_coupling(H_SE, tau)     U_SE = exp(-i H_SE tau)
ising_coupling(n_sys, n_E, J, tau)  H_SE = J Σ_k Z_k^S Z_0^E
xx_coupling(n_sys, n_E, g, tau)     H_SE = g Σ_k (X_k^S X_0^E + Y_k^S Y_0^E)
random_unitary(n_sys, n_E, ...)     U_SE = exp(-i strength H_rand)
partial_swap(n_sys, n_E, theta)     partial SWAP between S0 and E0
cnot_env_coupling(n_sys, n_E, ...)  CNOT_(S0→E0) then Ry on E0
kraus_from_USE(U_SE, n_sys, n_E)    Kraus operators from U_SE (env starts in |0⟩)

Convention: system qubits are the leftmost Kronecker factors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Callable

import numpy as np
from scipy.linalg import expm

from .operators import I2, X, Y, Z, lift


# ---------------------------------------------------------------------------
# Decay functions (matching stim/nonmarkovian convention)
# ---------------------------------------------------------------------------

def poly_decay(r: np.ndarray, A: float, q: float, n: float) -> np.ndarray:
    """
    Polynomial interaction decay: p(Δt) = A·q / Δt^n.

    Parameters
    ----------
    r : array-like of positive floats
        Time separations Δt = t2 − t1.
    A : float
        Overall amplitude.
    q : float
        Prefactor (often a base probability or qubit factor).
    n : float
        Polynomial exponent (larger n → faster decay).
    """
    return A * q / np.power(np.asarray(r, dtype=float), n)


def exp_decay(r: np.ndarray, A: float, q: float, n: float) -> np.ndarray:
    """
    Exponential interaction decay: p(Δt) = A·q / n^Δt.

    Parameters
    ----------
    r : array-like of positive floats
        Time separations Δt = t2 − t1.
    A : float
        Overall amplitude.
    q : float
        Prefactor (often a base probability or qubit factor).
    n : float
        Base of the exponential (n > 1 gives decay; larger n → faster decay).
    """
    return A * q / np.power(float(n), np.asarray(r, dtype=float))


# ---------------------------------------------------------------------------
# Noise model classes
# ---------------------------------------------------------------------------

class NoiseModel:
    """Abstract base for noise models passed to the simulation engine."""
    #: Number of environment qubits (0 for system-only channels).
    n_E: int = 0


class UnitarySENoise(NoiseModel):
    """
    Non-Markovian noise via a persistent system-environment coupling.

    The same U_SE is applied after every logical gate.  The environment
    register is kept across QEC cycles so quantum information can leak between
    them (non-Markovian memory).

    For the Markovian reference, pass reset_E=True to run_logical_rb.
    After each QEC cycle the engine will trace out the environment and
    reinitialise it to |0⟩_E (deterministic partial-trace reset, no sampling).

    Parameters
    ----------
    U_SE : array, shape (2**(n_sys+n_E), 2**(n_sys+n_E))
    n_E : int  — number of environment qubits (>= 1)
    """
    def __init__(self, U_SE: np.ndarray, n_E: int) -> None:
        self.U_SE = np.asarray(U_SE, dtype=complex)
        self.n_E  = int(n_E)


class MarkovianKraus(NoiseModel):
    """
    Markovian CPTP noise: the same Kraus channel every cycle, no environment.

        Λ(ρ) = Σ_k K_k ρ K_k†

    Use kraus_from_USE() to extract Kraus operators from an SE coupling
    unitary.  Single-qubit error channels (depolarising, amplitude damping,
    etc.) can be lifted to the physical code space via kron_list() and
    pauli_string() from operators.py.

    Parameters
    ----------
    kraus_ops : list of arrays, each shape (dim_S, dim_S)
        Must satisfy Σ_k K_k† K_k = I_S.
    """
    n_E = 0

    def __init__(self, kraus_ops: Sequence[np.ndarray]) -> None:
        self.kraus_ops = [np.asarray(K, dtype=complex) for K in kraus_ops]


class TimeVaryingKraus(NoiseModel):
    """
    Time-structured CPTP noise without a quantum environment.

    At gate cycle t (0-indexed) the channel is

        Λ_t(ρ) = Σ_k K_{t,k} ρ K_{t,k}†.

    The list wraps periodically, so you can model e.g. a two-cycle pattern
    by providing two Kraus operator sets.

    Parameters
    ----------
    kraus_ops_list : list of lists of arrays
        Outer index = time step t.  Each inner list is the set of Kraus
        operators K_{t,k}, shape (dim_S, dim_S).
    """
    n_E = 0

    def __init__(self, kraus_ops_list: Sequence[Sequence[np.ndarray]]) -> None:
        self.kraus_ops_list = [
            [np.asarray(K, dtype=complex) for K in ops]
            for ops in kraus_ops_list
        ]


class AncillaBitFlipNoise(NoiseModel):
    """
    Ancilla (syndrome-extraction) bit-flip errors wrapping any base noise model.

    In a real device, each stabilizer is measured via an ancilla qubit.  A
    bit-flip on that ancilla flips the readout, so the decoder receives a
    corrupted syndrome and applies the wrong recovery operator.

    Because the simulation marginalises over all syndromes analytically (no
    ancilla qubits are explicitly tracked), this is implemented by modifying
    the QEC Kraus channel.  For true syndrome s and ancilla error pattern e
    (each bit independently flipped with probability p_anc), the Kraus
    operator becomes

        K_{s,e} = sqrt(p(e)) R(s ⊕ e) Π_s,

    where p(e) = Π_k p_anc^{e_k} (1-p_anc)^{1-e_k}.  The data-qubit noise
    channel is delegated entirely to the wrapped ``base`` model.

    Parameters
    ----------
    base : NoiseModel
        Underlying data-qubit noise model (UnitarySENoise, MarkovianKraus, or
        TimeVaryingKraus).
    p_anc : float in [0, 0.5]
        Per-ancilla bit-flip probability during syndrome readout.
    """

    def __init__(self, base: NoiseModel, p_anc: float) -> None:
        if not isinstance(base, NoiseModel):
            raise TypeError("base must be a NoiseModel instance")
        p_anc = float(p_anc)
        if not (0.0 <= p_anc <= 0.5):
            raise ValueError(f"p_anc must be in [0, 0.5], got {p_anc}")
        self.base  = base
        self.p_anc = p_anc
        self.n_E   = base.n_E


# ---------------------------------------------------------------------------
# Temporally correlated noise (pairwise / streaky)
# ---------------------------------------------------------------------------

class PairwiseCorrelatedNoise(NoiseModel):
    """
    Pairwise temporally correlated noise for LRB.

    Adapted from Kam et al. arXiv:2410.23779 for the density-matrix LRB
    engine.  The "rounds" in that work become gate cycles here.

    At each ordered pair of gate cycles (t1, t2) with t1 < t2, an
    independent joint error event fires with probability p(t2 − t1), where p
    is given by interaction_func.  When a pair fires, error_kraus is applied
    at both t1 and t2.  Multiple overlapping events compose via XOR (an even
    number at the same step cancels).

    Because the error pattern at each step depends on which pairs fired (a
    joint distribution over all steps), this cannot be expressed as
    independent per-step Kraus channels.  The engine evaluates the exact
    survival probability by averaging over n_mc independently sampled error
    configurations (Monte Carlo over correlation patterns, not over quantum
    outcomes).

    Parameters
    ----------
    interaction_func : callable  f(distances) → probabilities
        Input: 1-D array of positive integer time separations Δt = t2 − t1.
        Output: array of pair-fire probabilities, same shape.
    error_kraus : list of arrays, shape (dim_S, dim_S)
        Kraus operators for a single error event on the full system space.
    n_mc : int
        Monte Carlo samples per gate sequence (default 500).
    """
    n_E = 0

    def __init__(
        self,
        interaction_func: Callable,
        error_kraus: Sequence[np.ndarray],
        n_mc: int = 500,
    ) -> None:
        self.interaction_func = interaction_func
        self.error_kraus      = [np.asarray(K, dtype=complex) for K in error_kraus]
        self.n_mc             = int(n_mc)

    def _sample_step_errors(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a binary error flag for each of the m+1 gate cycles.

        For each ordered pair (t1, t2) with t1 < t2, the pair fires
        independently with probability p(t2 − t1).  Errors accumulate via
        XOR: an even number of firing events at the same step cancels out.

        Returns integer array of shape (m+1,); nonzero → error at that step.
        """
        n_steps = m + 1
        # Build all ordered pairs (t1, t2) with t1 < t2 in a single pass.
        # There are n_steps*(n_steps-1)/2 such pairs.
        t1s = np.array([t1 for t1 in range(n_steps) for t2 in range(t1 + 1, n_steps)], dtype=int)
        t2s = np.array([t2 for t1 in range(n_steps) for t2 in range(t1 + 1, n_steps)], dtype=int)

        if len(t1s) == 0:
            return np.zeros(n_steps, dtype=int)

        distances = (t2s - t1s).astype(float)
        probs     = np.clip(np.asarray(self.interaction_func(distances), dtype=float), 0.0, 1.0)
        fired     = rng.random(len(probs)) < probs

        step_errors = np.zeros(n_steps, dtype=int)
        for t1, t2, f in zip(t1s, t2s, fired):
            if f:
                step_errors[t1] ^= 1
                step_errors[t2] ^= 1
        return step_errors

    def calc_marginals_per_cycle(self, m: int) -> np.ndarray:
        """
        Marginal error probability at each of the m+1 gate cycles.

        The marginal at step t is P(step t receives an error), averaged over
        all possible pair-firing configurations.  Uses the XOR convolution
        formula (Kam et al., Appendix A):

            p_t = ½ (1 − ∏_s (1 − 2 p(|t − s|)))

        where the product runs over all other steps s.

        Note: the marginals depend on m because the set of pairs involving
        step t grows with the total sequence length.  Boundary steps (t=0,
        t=m) have only one partner at each distance; interior steps have two,
        so interior marginals are generally higher.

        Returns
        -------
        marginals : ndarray of shape (m+1,)
        """
        n_steps = m + 1
        marginals = np.zeros(n_steps)
        for t in range(n_steps):
            distances = np.array(
                [float(abs(t - s)) for s in range(n_steps) if s != t],
                dtype=float,
            )
            probs = np.clip(
                np.asarray(self.interaction_func(distances), dtype=float),
                0.0, 1.0,
            )
            marginals[t] = 0.5 * float(1.0 - np.prod(1.0 - 2.0 * probs))
        return np.clip(marginals, 0.0, 1.0)

    def gen_markovian_baseline(self, m: int) -> "TimeVaryingKraus":
        """
        Build a Markovian baseline noise model for a sequence of length m.

        Returns a TimeVaryingKraus whose step-t Kraus channel is a convex
        mixture with the same marginal error probability as this correlated
        model:

            Λ_t(ρ) = (1 − p_t) ρ + p_t Λ_error(ρ)

        where p_t = calc_marginals_per_cycle(m)[t].

        Use this to disentangle the effect of temporal correlations from the
        effect of the per-cycle error rates: compare run_logical_rb results
        from the correlated model against this independent baseline.

        Important: the returned TimeVaryingKraus has period m+1 and is only
        valid for rb_sequence_survival(m, ...) or run_logical_rb([m], ...).
        Build a fresh baseline for each sequence length.
        """
        marginals = self.calc_marginals_per_cycle(m)
        dim_s = self.error_kraus[0].shape[0]
        I_s   = np.eye(dim_s, dtype=complex)
        kraus_ops_list = []
        for p_t in marginals:
            p_t  = float(p_t)
            ops  = [np.sqrt(1.0 - p_t) * I_s]
            ops += [np.sqrt(p_t) * E_k for E_k in self.error_kraus]
            kraus_ops_list.append(ops)
        return TimeVaryingKraus(kraus_ops_list)


class StreakCorrelatedNoise(NoiseModel):
    """
    Streaky temporally correlated noise for LRB.

    Adapted from Kam et al. arXiv:2410.23779.  A burst event spanning gate
    cycles [t1, t2] fires independently with probability p(t2 − t1).  When
    a burst fires, each cycle t in [t1, t2] independently receives an error
    with probability 0.5 (maximal mixing within the streak, matching the
    stim StreakNoiseModel._int_to_bool convention).  Overlapping bursts
    compose via XOR at each step.

    Parameters
    ----------
    interaction_func : callable  f(distances) → probabilities
    error_kraus : list of arrays, shape (dim_S, dim_S)
    n_mc : int
        Monte Carlo samples per gate sequence (default 500).
    """
    n_E = 0

    def __init__(
        self,
        interaction_func: Callable,
        error_kraus: Sequence[np.ndarray],
        n_mc: int = 500,
    ) -> None:
        self.interaction_func = interaction_func
        self.error_kraus      = [np.asarray(K, dtype=complex) for K in error_kraus]
        self.n_mc             = int(n_mc)

    def _sample_step_errors(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a binary error flag for each of the m+1 gate cycles.

        Each ordered pair (t1, t2) is a potential burst.  If a burst fires,
        every step in [t1, t2] independently receives an error with probability
        0.5.  Contributions from multiple bursts covering the same step are
        XOR'd (an even number of error contributions cancels out).

        Returns integer array of shape (m+1,); nonzero → error at that step.
        """
        n_steps = m + 1
        # All ordered pairs (t1, t2) with t1 < t2 — one potential burst each.
        t1s = np.array([t1 for t1 in range(n_steps) for t2 in range(t1 + 1, n_steps)], dtype=int)
        t2s = np.array([t2 for t1 in range(n_steps) for t2 in range(t1 + 1, n_steps)], dtype=int)

        if len(t1s) == 0:
            return np.zeros(n_steps, dtype=int)

        distances = (t2s - t1s).astype(float)
        probs     = np.clip(np.asarray(self.interaction_func(distances), dtype=float), 0.0, 1.0)
        fired     = rng.random(len(probs)) < probs

        # Count how many fired bursts cover each step.
        burst_counts = np.zeros(n_steps, dtype=int)
        for t1, t2, f in zip(t1s, t2s, fired):
            if f:
                burst_counts[t1:t2 + 1] += 1

        # Each active burst independently contributes a Bernoulli(0.5) error;
        # XOR the contributions across all bursts covering the same step.
        step_errors = np.zeros(n_steps, dtype=int)
        for t in range(n_steps):
            if burst_counts[t] > 0:
                bits = rng.integers(0, 2, size=burst_counts[t])  # Bernoulli(0.5)
                step_errors[t] = int(np.sum(bits) % 2)
        return step_errors

    def calc_marginals_per_cycle(self, m: int) -> np.ndarray:
        """
        Marginal error probability at each of the m+1 gate cycles.

        A burst spanning [t1, t2] fires with probability p(t2−t1) and, if
        it fires, each step in the interval independently gets an error with
        probability 0.5.  The unconditional probability that burst (t1,t2)
        causes an error at step t (given t1 ≤ t ≤ t2) is therefore
        0.5 · p(t2−t1).  Collecting all bursts covering t and applying the
        XOR convolution formula:

            p_t = ½ (1 − ∏_{(t1,t2): t1≤t≤t2} (1 − p(t2 − t1)))

        Returns
        -------
        marginals : ndarray of shape (m+1,)
        """
        n_steps = m + 1
        # Precompute all (t1, t2) pair probabilities
        pairs = [
            (t1, t2)
            for t1 in range(n_steps)
            for t2 in range(t1 + 1, n_steps)
        ]
        if not pairs:
            return np.zeros(n_steps)

        distances   = np.array([float(t2 - t1) for t1, t2 in pairs], dtype=float)
        burst_probs = np.clip(
            np.asarray(self.interaction_func(distances), dtype=float),
            0.0, 1.0,
        )

        marginals = np.zeros(n_steps)
        for t in range(n_steps):
            # For each burst (t1, t2) covering step t, the unconditional
            # probability that this burst contributes an error at t is
            # q_i = P(burst fires) × P(error | fires) = p(t2−t1) × 0.5.
            covering = np.array(
                [0.5 * bp for (t1, t2), bp in zip(pairs, burst_probs) if t1 <= t <= t2],
                dtype=float,
            )
            if len(covering) > 0:
                # Apply XOR convolution formula p_XOR = ½(1 − ∏(1 − 2 q_i))
                # with q_i = covering[i] = 0.5·p_burst_i.  The two factors of
                # 0.5 cancel: 2·q_i = p_burst_i, so this equals
                # ½(1 − ∏(1 − p_burst_i)), matching the docstring formula.
                marginals[t] = 0.5 * float(1.0 - np.prod(1.0 - 2.0 * covering))
        return np.clip(marginals, 0.0, 1.0)

    def gen_markovian_baseline(self, m: int) -> "TimeVaryingKraus":
        """
        Build a Markovian baseline noise model for a sequence of length m.

        Returns a TimeVaryingKraus whose step-t Kraus channel matches the
        marginal error probability of this streaky model:

            Λ_t(ρ) = (1 − p_t) ρ + p_t Λ_error(ρ),

        where p_t = calc_marginals_per_cycle(m)[t].

        Use this to isolate the effect of temporal correlations: compare
        run_logical_rb with this model against the correlated model — any
        difference is due to burst structure, not per-cycle error rates.

        Important: the returned TimeVaryingKraus has period m+1 and is only
        valid for rb_sequence_survival(m, ...) or run_logical_rb([m], ...).
        Build a fresh baseline for each sequence length.
        """
        marginals = self.calc_marginals_per_cycle(m)
        dim_s = self.error_kraus[0].shape[0]
        I_s   = np.eye(dim_s, dtype=complex)
        kraus_ops_list = []
        for p_t in marginals:
            p_t  = float(p_t)
            ops  = [np.sqrt(1.0 - p_t) * I_s]
            ops += [np.sqrt(p_t) * E_k for E_k in self.error_kraus]
            kraus_ops_list.append(ops)
        return TimeVaryingKraus(kraus_ops_list)


# Convenience subclasses matching the stim model naming

class PairwisePolyNoise(PairwiseCorrelatedNoise):
    """Pairwise, polynomial decay  p(Δt) = A·q / Δt^n"""
    def __init__(self, A: float, q: float, n: float,
                 error_kraus: Sequence[np.ndarray], n_mc: int = 500) -> None:
        super().__init__(
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            error_kraus=error_kraus, n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class PairwiseExpNoise(PairwiseCorrelatedNoise):
    """Pairwise, exponential decay  p(Δt) = A·q / n^Δt"""
    def __init__(self, A: float, q: float, n: float,
                 error_kraus: Sequence[np.ndarray], n_mc: int = 500) -> None:
        super().__init__(
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            error_kraus=error_kraus, n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class StreakPolyNoise(StreakCorrelatedNoise):
    """Streaky, polynomial decay  p(burst length Δt) = A·q / Δt^n"""
    def __init__(self, A: float, q: float, n: float,
                 error_kraus: Sequence[np.ndarray], n_mc: int = 500) -> None:
        super().__init__(
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            error_kraus=error_kraus, n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class StreakExpNoise(StreakCorrelatedNoise):
    """Streaky, exponential decay  p(burst length Δt) = A·q / n^Δt"""
    def __init__(self, A: float, q: float, n: float,
                 error_kraus: Sequence[np.ndarray], n_mc: int = 500) -> None:
        super().__init__(
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            error_kraus=error_kraus, n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_qubit_counts(
    n_sys: int,
    n_E: int,
    *,
    require_environment: bool = True,
) -> None:
    """Validate the numbers of system and environment qubits."""
    if (
        not isinstance(n_sys, (int, np.integer))
        or isinstance(n_sys, bool)
    ):
        raise TypeError("n_sys must be an integer")

    if (
        not isinstance(n_E, (int, np.integer))
        or isinstance(n_E, bool)
    ):
        raise TypeError("n_E must be an integer")

    if n_sys < 1:
        raise ValueError(f"n_sys must be at least 1, got {n_sys}")

    minimum_environment = 1 if require_environment else 0

    if n_E < minimum_environment:
        raise ValueError(
            f"n_E must be at least {minimum_environment}, got {n_E}"
        )


def _validate_real_finite(value: float, name: str) -> float:
    """Return value as a finite real float."""
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


# ---------------------------------------------------------------------------
# General Hamiltonian evolution
# ---------------------------------------------------------------------------

def hamiltonian_coupling(
    H_SE: np.ndarray,
    tau: float,
    *,
    hermitian_tolerance: float = 1e-10,
) -> np.ndarray:
    """
    Return

        U_SE = exp(-i H_SE tau).

    H_SE must be a finite, square Hermitian matrix.
    """
    H_SE = np.asarray(H_SE, dtype=complex)
    tau = _validate_real_finite(tau, "tau")

    if H_SE.ndim != 2 or H_SE.shape[0] != H_SE.shape[1]:
        raise ValueError(
            f"H_SE must be square, got shape {H_SE.shape}"
        )

    if not np.all(np.isfinite(H_SE)):
        raise ValueError("H_SE contains nonfinite entries")

    if not np.allclose(
        H_SE,
        H_SE.conj().T,
        atol=hermitian_tolerance,
        rtol=0.0,
    ):
        error = np.linalg.norm(H_SE - H_SE.conj().T)
        raise ValueError(
            f"H_SE must be Hermitian; ||H-H†|| = {error:.3e}"
        )

    return expm(-1j * tau * H_SE)


# ---------------------------------------------------------------------------
# Structured couplings
# ---------------------------------------------------------------------------

def ising_coupling(
    n_sys: int,
    n_E: int,
    J: float,
    tau: float,
) -> np.ndarray:
    """
    Construct the Ising system-environment coupling

        H_SE = J sum_k Z_k^S Z_0^E,

    and return U_SE = exp(-i H_SE tau).

    Every system qubit couples to environment qubit 0.
    """
    _validate_qubit_counts(n_sys, n_E)

    J = _validate_real_finite(J, "J")
    tau = _validate_real_finite(tau, "tau")

    n_total = n_sys + n_E
    dim = 2**n_total

    H_SE = np.zeros((dim, dim), dtype=complex)
    Z_E0 = lift(Z, n_sys, n_total)

    for k in range(n_sys):
        Z_k = lift(Z, k, n_total)
        H_SE += J * (Z_k @ Z_E0)

    return hamiltonian_coupling(H_SE, tau)


def xx_coupling(
    n_sys: int,
    n_E: int,
    g: float,
    tau: float,
) -> np.ndarray:
    """
    Construct the exchange coupling

        H_SE = g sum_k (X_k^S X_0^E + Y_k^S Y_0^E),

    and return U_SE = exp(-i H_SE tau).
    """
    _validate_qubit_counts(n_sys, n_E)

    g = _validate_real_finite(g, "g")
    tau = _validate_real_finite(tau, "tau")

    n_total = n_sys + n_E
    dim = 2**n_total

    H_SE = np.zeros((dim, dim), dtype=complex)

    X_E0 = lift(X, n_sys, n_total)
    Y_E0 = lift(Y, n_sys, n_total)

    for k in range(n_sys):
        X_k = lift(X, k, n_total)
        Y_k = lift(Y, k, n_total)

        H_SE += g * (
            X_k @ X_E0
            + Y_k @ Y_E0
        )

    return hamiltonian_coupling(H_SE, tau)


# ---------------------------------------------------------------------------
# Random coupling
# ---------------------------------------------------------------------------

def random_unitary(
    n_sys: int,
    n_E: int,
    strength: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Construct

        U_SE = exp(-i strength H_rand),

    where H_rand is a random Hermitian matrix normalized so that

        ||H_rand||_2 = 1.

    This normalization makes `strength` comparable across different
    Hilbert-space dimensions.

    Exactly one of `rng` and `seed` may be supplied.
    """
    _validate_qubit_counts(n_sys, n_E)

    strength = _validate_real_finite(strength, "strength")

    if strength < 0:
        raise ValueError(
            f"strength must be nonnegative, got {strength}"
        )

    if rng is not None and seed is not None:
        raise ValueError("Specify either rng or seed, not both")

    if rng is None:
        rng = np.random.default_rng(seed)

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    dim = 2 ** (n_sys + n_E)

    A = (
        rng.standard_normal((dim, dim))
        + 1j * rng.standard_normal((dim, dim))
    )

    H_rand = (A + A.conj().T) / 2

    # Normalize the spectral norm so that the largest eigenphase magnitude
    # is bounded by `strength`.
    spectral_norm = np.linalg.norm(H_rand, ord=2)

    if spectral_norm <= 1e-14:
        return np.eye(dim, dtype=complex)

    H_rand /= spectral_norm

    return expm(-1j * strength * H_rand)


# ---------------------------------------------------------------------------
# Partial SWAP
# ---------------------------------------------------------------------------

def partial_swap(
    n_sys: int,
    n_E: int,
    theta: float,
) -> np.ndarray:
    """
    Construct a partial SWAP between S0 and E0:

        U_SE = cos(theta) I + i sin(theta) SWAP_(S0,E0)
             = exp(+i theta SWAP_(S0,E0)).

    Other system and environment qubits are unaffected.

    Note that Hamiltonian evolution exp(-i theta SWAP) would instead have
    a minus sign in front of the imaginary term.
    """
    _validate_qubit_counts(n_sys, n_E)
    theta = _validate_real_finite(theta, "theta")

    n_total = n_sys + n_E
    dim = 2**n_total

    identity = np.eye(dim, dtype=complex)

    # SWAP_ab = 1/2 (I + X_a X_b + Y_a Y_b + Z_a Z_b).
    swap = np.zeros((dim, dim), dtype=complex)

    for P in (I2, X, Y, Z):
        P_S0 = lift(P, 0, n_total)
        P_E0 = lift(P, n_sys, n_total)
        swap += P_S0 @ P_E0

    swap /= 2

    return (
        np.cos(theta) * identity
        + 1j * np.sin(theta) * swap
    )


# ---------------------------------------------------------------------------
# CNOT coupling
# ---------------------------------------------------------------------------

def cnot_env_coupling(
    n_sys: int,
    n_E: int,
    theta_env: float,
) -> np.ndarray:
    """
    Apply CNOT with S0 as control and E0 as target, followed by
    Ry(theta_env) on E0.

    Therefore,

        U_SE = Ry_E0(theta_env) CNOT_(S0 -> E0).
    """
    _validate_qubit_counts(n_sys, n_E)
    theta_env = _validate_real_finite(theta_env, "theta_env")

    n_total = n_sys + n_E
    dim = 2**n_total

    control = 0
    target = n_sys

    control_mask = 1 << (n_total - 1 - control)
    target_mask = 1 << (n_total - 1 - target)

    # Build CNOT directly as a permutation matrix.
    CNOT = np.zeros((dim, dim), dtype=complex)

    for input_state in range(dim):
        output_state = input_state

        if input_state & control_mask:
            output_state ^= target_mask

        CNOT[output_state, input_state] = 1.0

    cosine = np.cos(theta_env / 2)
    sine = np.sin(theta_env / 2)

    Ry = np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=complex,
    )

    Ry_E0 = lift(Ry, target, n_total)

    # Rightmost gate acts first: CNOT first, then Ry.
    return Ry_E0 @ CNOT


# ---------------------------------------------------------------------------
# Kraus representation
# ---------------------------------------------------------------------------

def kraus_from_USE(
    U_SE: np.ndarray,
    n_sys: int,
    n_E: int,
    *,
    unitary_tolerance: float = 1e-9,
) -> list[np.ndarray]:
    """
    Decompose U_SE into Kraus operators on S, assuming that E starts in

        |0...0>_E.

    The Kraus operators are

        K_k = <k|_E U_SE |0...0>_E,

    or, in components,

        K_k[i,j] = <i,k| U_SE |j,0>.

    System indices are the leading tensor indices.
    """
    _validate_qubit_counts(
        n_sys,
        n_E,
        require_environment=False,
    )

    dim_s = 2**n_sys
    dim_e = 2**n_E
    dim_total = dim_s * dim_e

    U_SE = np.asarray(U_SE, dtype=complex)

    expected_shape = (dim_total, dim_total)

    if U_SE.shape != expected_shape:
        raise ValueError(
            f"U_SE must have shape {expected_shape}, "
            f"got {U_SE.shape}"
        )

    if not np.all(np.isfinite(U_SE)):
        raise ValueError("U_SE contains nonfinite entries")

    identity = np.eye(dim_total, dtype=complex)

    if not np.allclose(
        U_SE.conj().T @ U_SE,
        identity,
        atol=unitary_tolerance,
        rtol=0.0,
    ):
        error = np.linalg.norm(
            U_SE.conj().T @ U_SE - identity
        )
        raise ValueError(
            f"U_SE must be unitary; ||U†U-I|| = {error:.3e}"
        )

    # Tensor indices:
    #
    #     U_tensor[system_out, environment_out,
    #              system_in,  environment_in].
    U_tensor = U_SE.reshape(
        dim_s,
        dim_e,
        dim_s,
        dim_e,
    )

    kraus = [
        U_tensor[:, k, :, 0].copy()
        for k in range(dim_e)
    ]

    # Numerical verification of trace preservation:
    #
    #     sum_k K_k† K_k = I_S.
    completeness = np.zeros((dim_s, dim_s), dtype=complex)

    for K in kraus:
        completeness += K.conj().T @ K

    identity_s = np.eye(dim_s, dtype=complex)

    if not np.allclose(
        completeness,
        identity_s,
        atol=unitary_tolerance,
        rtol=0.0,
    ):
        error = np.linalg.norm(completeness - identity_s)
        raise RuntimeError(
            "Extracted Kraus operators are not trace preserving; "
            f"||sum(K†K)-I|| = {error:.3e}"
        )

    return kraus