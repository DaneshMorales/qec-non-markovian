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
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros(n_steps, dtype=int)
        probs = np.clip(self.interaction_func((t2s - t1s).astype(float)), 0.0, 1.0)
        fired = rng.random(len(probs)) < probs
        # XOR scatter: count how many fired pairs land on each step, mod 2.
        return (
            np.bincount(t1s[fired], minlength=n_steps)
            + np.bincount(t2s[fired], minlength=n_steps)
        ) % 2

    def _sample_step_errors_batch(
        self, m: int, n_mc: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample n_mc error patterns simultaneously.

        Returns (n_mc, m+1) int8 array; 1 = error fires at that (sample, step).
        Uses a single (n_mc × n_pairs) @ (n_pairs × n_steps) matrix multiply.
        """
        n_steps = m + 1
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros((n_mc, n_steps), dtype=np.int8)
        probs = np.clip(self.interaction_func((t2s - t1s).astype(float)), 0.0, 1.0)
        fired = (rng.random((n_mc, len(probs))) < probs).view(np.int8)  # (n_mc, n_pairs)
        ind = np.zeros((len(t1s), n_steps), dtype=np.int8)
        ind[np.arange(len(t1s)), t1s] = 1
        ind[np.arange(len(t2s)), t2s] = 1
        return (fired @ ind % 2).astype(np.int8)  # (n_mc, n_steps)

    def calc_marginals_per_cycle(self, m: int) -> np.ndarray:
        """
        Marginal error probability at each of the m+1 gate cycles.

        The marginal at step t is P(step t receives an error), averaged over
        all possible pair-firing configurations.  Uses the XOR convolution
        formula (Kam et al., Appendix A):

            p_t = ½ (1 − ∏_s (1 − 2 p(|t − s|)))

        where the product runs over all other steps s ≠ t.
        """
        n_steps = m + 1
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros(n_steps)
        dists = (t2s - t1s).astype(float)
        probs = np.clip(self.interaction_func(dists), 0.0, 1.0)  # (n_pairs,)

        # For step t, the relevant pairs are those with t1s[k] == t or t2s[k] == t.
        # Build a (n_steps, n_pairs) indicator and compute log-product in one pass.
        # indicator[t, k] = 1 iff pair k involves step t.
        indicator = np.zeros((n_steps, len(t1s)), dtype=bool)
        indicator[t1s, np.arange(len(t1s))] = True
        indicator[t2s, np.arange(len(t2s))] = True
        # log(1 - 2p) for each pair; 0 where not involved
        log_terms = np.where(indicator, np.log(np.clip(1.0 - 2.0 * probs, 1e-15, None)), 0.0)
        marginals = 0.5 * (1.0 - np.exp(log_terms.sum(axis=1)))
        return np.clip(marginals, 0.0, 1.0)

    def apply_error(self, rho: np.ndarray) -> np.ndarray:
        """Apply the error channel Λ_error(ρ) to a single density matrix."""
        out = np.zeros_like(rho)
        for K in self.error_kraus:
            out += K @ rho @ K.conj().T
        return out

    def apply_error_batch(self, rho_batch: np.ndarray) -> np.ndarray:
        """Apply Λ_error to a batch (n, d, d) using numpy broadcasting."""
        out = np.zeros_like(rho_batch)
        for K in self.error_kraus:
            out += K @ rho_batch @ K.conj().T
        return out

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
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros(n_steps, dtype=int)
        probs = np.clip(self.interaction_func((t2s - t1s).astype(float)), 0.0, 1.0)
        fired = rng.random(len(probs)) < probs
        if not fired.any():
            return np.zeros(n_steps, dtype=int)

        # Vectorised segment coverage via a cumsum difference array — no Python
        # loop over pairs.  diff[t1] += 1 and diff[t2+1] -= 1 for each fired
        # burst; cumsum gives burst_counts[t] = number of fired bursts covering t.
        t1f, t2f = t1s[fired], t2s[fired]
        diff = np.zeros(n_steps + 1, dtype=np.int32)
        np.add.at(diff, t1f, 1)
        np.add.at(diff, t2f + 1, -1)
        burst_counts = np.cumsum(diff)[:n_steps]

        # XOR of n independent Bernoulli(0.5) is still Bernoulli(0.5) for n > 0.
        # Sample all covered steps at once.
        covered = burst_counts > 0
        step_errors = np.zeros(n_steps, dtype=int)
        step_errors[covered] = rng.integers(0, 2, size=int(covered.sum()))
        return step_errors

    def _sample_step_errors_batch(
        self, m: int, n_mc: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample n_mc streak error patterns simultaneously.

        Returns (n_mc, m+1) int8 array; 1 = error at that (sample, step).

        Uses a cumsum difference-array approach fully vectorised over n_mc:
        coverage counts are computed via a (n_mc × n_pairs) @ (n_pairs × n_steps+1)
        matrix multiply, then a cumsum along the step axis.
        """
        n_steps = m + 1
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros((n_mc, n_steps), dtype=np.int8)
        probs = np.clip(self.interaction_func((t2s - t1s).astype(float)), 0.0, 1.0)
        fired = rng.random((n_mc, len(probs))) < probs  # (n_mc, n_pairs)
        ind_diff = np.zeros((len(t1s), n_steps + 1), dtype=np.int8)
        ind_diff[np.arange(len(t1s)), t1s]     =  1
        ind_diff[np.arange(len(t2s)), t2s + 1] = -1
        batch_diff    = fired.astype(np.int8) @ ind_diff        # (n_mc, n_steps+1)
        burst_counts  = np.cumsum(batch_diff, axis=1)[:, :n_steps]
        covered       = burst_counts > 0
        step_errors   = np.zeros((n_mc, n_steps), dtype=np.int8)
        step_errors[covered] = rng.integers(0, 2, size=int(covered.sum()), dtype=np.int8)
        return step_errors

    def calc_marginals_per_cycle(self, m: int) -> np.ndarray:
        """
        Marginal error probability at each of the m+1 gate cycles.

        A burst spanning [t1, t2] fires with probability p(t2−t1) and, if
        it fires, each step in the interval independently gets an error with
        probability 0.5.  Collecting all bursts covering t and applying the
        XOR convolution formula:

            p_t = ½ (1 − ∏_{(t1,t2): t1≤t≤t2} (1 − p(t2 − t1)))
        """
        n_steps = m + 1
        t1s, t2s = np.triu_indices(n_steps, k=1)
        if len(t1s) == 0:
            return np.zeros(n_steps)
        probs = np.clip(self.interaction_func((t2s - t1s).astype(float)), 0.0, 1.0)

        # coverage[t, k] = True iff burst k covers step t (t1s[k] <= t <= t2s[k]).
        # Build via a cumsum difference-array: broadcast over all t at once.
        t_axis = np.arange(n_steps)[:, None]   # (n_steps, 1)
        coverage = (t1s[None, :] <= t_axis) & (t_axis <= t2s[None, :])  # (n_steps, n_pairs)
        log_terms = np.where(coverage, np.log(np.clip(1.0 - probs, 1e-15, None)), 0.0)
        marginals = 0.5 * (1.0 - np.exp(log_terms.sum(axis=1)))
        return np.clip(marginals, 0.0, 1.0)

    def apply_error(self, rho: np.ndarray) -> np.ndarray:
        """Apply the error channel Λ_error(ρ) to a single density matrix."""
        out = np.zeros_like(rho)
        for K in self.error_kraus:
            out += K @ rho @ K.conj().T
        return out

    def apply_error_batch(self, rho_batch: np.ndarray) -> np.ndarray:
        """Apply Λ_error to a batch (n, d, d) using numpy broadcasting."""
        out = np.zeros_like(rho_batch)
        for K in self.error_kraus:
            out += K @ rho_batch @ K.conj().T
        return out

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
# Fully-depolarizing correlated noise and its Markovian baseline
# ---------------------------------------------------------------------------

class PartialDepolarizingKraus(NoiseModel):
    """
    Time-varying partial depolarizing channel:  Λ_t(ρ) = (1-p_t)ρ + p_t · I/dim_s.

    Applied by the engine as a direct convex mixture — no Kraus operator loop
    needed — making it efficient even for large code spaces (Steane, etc.).
    Returned by FullyDepolarizingPairwiseNoise.gen_markovian_baseline() and
    FullyDepolarizingStreakNoise.gen_markovian_baseline() to serve as the
    marginalized independent comparison model.

    Parameters
    ----------
    p_list : sequence of float
        Per-cycle error probabilities p_t for t = 0, 1, ..., m.  Length
        should equal m+1 for a sequence of length m.  Applied cyclically
        (modulo len(p_list)), so a uniform list gives a stationary channel.
    dim_s : int
        System Hilbert-space dimension (2**n_code for an n_code-qubit code).
    """
    n_E = 0

    def __init__(self, p_list: Sequence[float], dim_s: int) -> None:
        self.p_list = [float(p) for p in p_list]
        self.dim_s  = int(dim_s)
        self._I_d   = np.eye(self.dim_s, dtype=complex) / self.dim_s


class DepolarizingPairwiseNoise(PairwiseCorrelatedNoise):
    """
    Pairwise temporally correlated noise with an n-qubit depolarizing error event.

    When a pairwise event fires at steps t1 and t2, the n-qubit depolarizing
    channel

        Λ(ρ) = (1 − p) ρ + p · I / 2**n_code

    is applied at both steps.  No Kraus operator loop is needed: the channel
    is computed as a direct convex mixture in O(dim_s²) time.

    This is the n-qubit generalization of the Class 0 depolarizing model in
    Kam et al. (arXiv:2410.23779, Sec. III A).

    The Markovian baseline gen_markovian_baseline(m) returns a
    PartialDepolarizingKraus that applies, at step t,

        Λ_t(ρ) = (1 − marginal_t · p) ρ + marginal_t · p · I / dim_s,

    i.e. the effective depolarizing rate is the marginal firing probability
    multiplied by the per-event strength p.

    Parameters
    ----------
    n_code : int
        Number of physical qubits in the code (e.g. 7 for SteaneCode).
    p : float in (0, 1]
        Depolarizing strength applied when a pairwise event fires.
        p = 1 gives the fully depolarizing channel Λ(ρ) = I/2**n_code.
    interaction_func : callable  f(distances) → probabilities
        Pair-fire probability as a function of time separations Δt.
    n_mc : int
        Monte Carlo samples per gate sequence (default 500).
    """

    def __init__(
        self,
        n_code: int,
        p: float,
        interaction_func: Callable,
        n_mc: int = 500,
    ) -> None:
        super().__init__(interaction_func, error_kraus=[], n_mc=n_mc)
        self.n_code = int(n_code)
        self.p      = float(p)
        self._dim_s = 2**self.n_code
        self._I_d   = np.eye(self._dim_s, dtype=complex) / self._dim_s

    def apply_error(self, rho: np.ndarray) -> np.ndarray:
        return (1.0 - self.p) * rho + self.p * self._I_d

    def apply_error_batch(self, rho_batch: np.ndarray) -> np.ndarray:
        """O(d²) partial-depolarizing update — no Kraus loop needed."""
        return (1.0 - self.p) * rho_batch + self.p * self._I_d

    def gen_markovian_baseline(self, m: int) -> PartialDepolarizingKraus:  # type: ignore[override]
        """
        Return a PartialDepolarizingKraus with per-cycle rates matching the
        marginals of this pairwise correlated model.

        The effective depolarizing rate at step t is marginals[t] * p, since
        the channel Λ_error fires with probability marginals[t] and has
        strength p:

            Λ_t^Markov(ρ) = (1 − marginals[t]) ρ + marginals[t] Λ_error(ρ)
                           = (1 − marginals[t]·p) ρ + marginals[t]·p · I/dim_s.
        """
        marginals = self.calc_marginals_per_cycle(m)
        p_eff = [float(mt) * self.p for mt in marginals]
        return PartialDepolarizingKraus(p_list=p_eff, dim_s=self._dim_s)


class DepolarizingStreakNoise(StreakCorrelatedNoise):
    """
    Streaky temporally correlated noise with an n-qubit depolarizing error event.

    When a burst event covers gate cycle t, the n-qubit depolarizing channel

        Λ(ρ) = (1 − p) ρ + p · I / 2**n_code

    is applied (independently per cycle within the burst, with 0.5 probability
    per cycle, matching the stim streaky convention).

    See DepolarizingPairwiseNoise for full documentation; the only difference
    is the temporal correlation structure (pairwise vs. burst).

    Parameters
    ----------
    n_code : int
    p : float in (0, 1]
        Depolarizing strength per error event.
    interaction_func : callable  f(distances) → probabilities
    n_mc : int
    """

    def __init__(
        self,
        n_code: int,
        p: float,
        interaction_func: Callable,
        n_mc: int = 500,
    ) -> None:
        super().__init__(interaction_func, error_kraus=[], n_mc=n_mc)
        self.n_code = int(n_code)
        self.p      = float(p)
        self._dim_s = 2**self.n_code
        self._I_d   = np.eye(self._dim_s, dtype=complex) / self._dim_s

    def apply_error(self, rho: np.ndarray) -> np.ndarray:
        return (1.0 - self.p) * rho + self.p * self._I_d

    def apply_error_batch(self, rho_batch: np.ndarray) -> np.ndarray:
        """O(d²) partial-depolarizing update — no Kraus loop needed."""
        return (1.0 - self.p) * rho_batch + self.p * self._I_d

    def gen_markovian_baseline(self, m: int) -> PartialDepolarizingKraus:  # type: ignore[override]
        """
        Return a PartialDepolarizingKraus with per-cycle effective rates
        marginals[t] * p (firing probability × depolarizing strength).
        """
        marginals = self.calc_marginals_per_cycle(m)
        p_eff = [float(mt) * self.p for mt in marginals]
        return PartialDepolarizingKraus(p_list=p_eff, dim_s=self._dim_s)


# Convenience subclasses for the common decay functions

class DepPolyPairwiseNoise(DepolarizingPairwiseNoise):
    """Depolarizing pairwise, polynomial decay  p_fire(Δt) = A·q / Δt^n"""
    def __init__(self, n_code: int, p: float, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code, p,
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class DepExpPairwiseNoise(DepolarizingPairwiseNoise):
    """Depolarizing pairwise, exponential decay  p_fire(Δt) = A·q / n^Δt"""
    def __init__(self, n_code: int, p: float, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code, p,
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class DepPolyStreakNoise(DepolarizingStreakNoise):
    """Depolarizing streaky, polynomial decay  p_fire(Δt) = A·q / Δt^n"""
    def __init__(self, n_code: int, p: float, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code, p,
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class DepExpStreakNoise(DepolarizingStreakNoise):
    """Depolarizing streaky, exponential decay  p_fire(Δt) = A·q / n^Δt"""
    def __init__(self, n_code: int, p: float, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code, p,
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


# Fully-depolarizing (p=1) special cases — kept as thin wrappers for convenience

class FullyDepolarizingPairwiseNoise(DepolarizingPairwiseNoise):
    """Pairwise correlated noise with p=1 (fully depolarizing). Use DepolarizingPairwiseNoise for p < 1."""
    def __init__(self, n_code: int, interaction_func: Callable, n_mc: int = 500) -> None:
        super().__init__(n_code, p=1.0, interaction_func=interaction_func, n_mc=n_mc)


class FullyDepolarizingStreakNoise(DepolarizingStreakNoise):
    """Streaky correlated noise with p=1 (fully depolarizing). Use DepolarizingStreakNoise for p < 1."""
    def __init__(self, n_code: int, interaction_func: Callable, n_mc: int = 500) -> None:
        super().__init__(n_code, p=1.0, interaction_func=interaction_func, n_mc=n_mc)


class FDPolyPairwiseNoise(FullyDepolarizingPairwiseNoise):
    """Fully depolarizing pairwise, polynomial decay  p_fire(Δt) = A·q / Δt^n"""
    def __init__(self, n_code: int, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code,
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class FDExpPairwiseNoise(FullyDepolarizingPairwiseNoise):
    """Fully depolarizing pairwise, exponential decay  p_fire(Δt) = A·q / n^Δt"""
    def __init__(self, n_code: int, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code,
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class FDPolyStreakNoise(FullyDepolarizingStreakNoise):
    """Fully depolarizing streaky, polynomial decay  p_fire(Δt) = A·q / Δt^n"""
    def __init__(self, n_code: int, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code,
            interaction_func=lambda r, _A=A, _q=q, _n=n: poly_decay(r, _A, _q, _n),
            n_mc=n_mc,
        )
        self.A, self.q, self.n = A, q, n


class FDExpStreakNoise(FullyDepolarizingStreakNoise):
    """Fully depolarizing streaky, exponential decay  p_fire(Δt) = A·q / n^Δt"""
    def __init__(self, n_code: int, A: float, q: float, n: float, n_mc: int = 500) -> None:
        super().__init__(
            n_code,
            interaction_func=lambda r, _A=A, _q=q, _n=n: exp_decay(r, _A, _q, _n),
            n_mc=n_mc,
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

        H_SE = J sum_k sum_j Z_k^S Z_j^E,

    and return U_SE = exp(-i H_SE tau).

    Every system qubit couples to every environment qubit (all-to-all ZZ).
    """
    _validate_qubit_counts(n_sys, n_E)

    J = _validate_real_finite(J, "J")
    tau = _validate_real_finite(tau, "tau")

    n_total = n_sys + n_E
    dim = 2**n_total

    H_SE = np.zeros((dim, dim), dtype=complex)

    Z_sys = [lift(Z, k, n_total) for k in range(n_sys)]
    Z_env = [lift(Z, n_sys + j, n_total) for j in range(n_E)]

    for Z_k in Z_sys:
        for Z_j in Z_env:
            H_SE += J * (Z_k @ Z_j)

    return hamiltonian_coupling(H_SE, tau)


def xx_coupling(
    n_sys: int,
    n_E: int,
    g: float,
    tau: float,
) -> np.ndarray:
    """
    Construct the exchange (XY) coupling

        H_SE = g sum_k sum_j (X_k^S X_j^E + Y_k^S Y_j^E),

    and return U_SE = exp(-i H_SE tau).

    Every system qubit couples to every environment qubit (all-to-all XY).
    """
    _validate_qubit_counts(n_sys, n_E)

    g = _validate_real_finite(g, "g")
    tau = _validate_real_finite(tau, "tau")

    n_total = n_sys + n_E
    dim = 2**n_total

    H_SE = np.zeros((dim, dim), dtype=complex)

    X_sys = [lift(X, k, n_total) for k in range(n_sys)]
    Y_sys = [lift(Y, k, n_total) for k in range(n_sys)]
    X_env = [lift(X, n_sys + j, n_total) for j in range(n_E)]
    Y_env = [lift(Y, n_sys + j, n_total) for j in range(n_E)]

    for X_k, Y_k in zip(X_sys, Y_sys):
        for X_j, Y_j in zip(X_env, Y_env):
            H_SE += g * (X_k @ X_j + Y_k @ Y_j)

    return hamiltonian_coupling(H_SE, tau)


def two_spin_coupling(
    n_sys: int,
    n_E: int,
    J: float,
    hx: float,
    hy: float,
    delta: float,
) -> np.ndarray:
    """
    Two-spin system-environment interaction (Eq. 24 of the paper):

        H = J Σ_k X_k^S X_j^E  +  h_x Σ_i X_i  +  h_y Σ_i Y_i

    The coupling term pairs every system qubit k with every environment qubit j
    via X_k X_j.  The transverse fields h_x and h_y act independently on every
    qubit (system and environment alike).

    Returns U_SE = exp(-i δ H).

    Parameters
    ----------
    n_sys  : int    Number of system qubits (3 for RepetitionCode).
    n_E    : int    Number of environment qubits (1 for the single-ancilla model).
    J      : float  XX coupling strength between system and environment qubits.
    hx     : float  Transverse X field applied to every qubit.
    hy     : float  Transverse Y field applied to every qubit.
    delta  : float  Evolution time δ  (U = exp(-i δ H)).

    Example (paper parameters, single qubit → single ancilla)
    ----------------------------------------------------------
    J=1.7, hx=1.47, hy=-1.05, delta=0.029475

    For the RepetitionCode (n_sys=3, n_E=1) these parameters carry over
    directly; the coupling and field operators simply extend over three system
    qubits instead of one.
    """
    _validate_qubit_counts(n_sys, n_E)
    J     = _validate_real_finite(J,     "J")
    hx    = _validate_real_finite(hx,    "hx")
    hy    = _validate_real_finite(hy,    "hy")
    delta = _validate_real_finite(delta, "delta")

    n_total = n_sys + n_E
    dim     = 2**n_total
    H_SE    = np.zeros((dim, dim), dtype=complex)

    X_sys = [lift(X, k,         n_total) for k in range(n_sys)]
    Y_sys = [lift(Y, k,         n_total) for k in range(n_sys)]
    X_env = [lift(X, n_sys + j, n_total) for j in range(n_E)]
    Y_env = [lift(Y, n_sys + j, n_total) for j in range(n_E)]

    # XX coupling: each system qubit couples to each environment qubit
    for X_k in X_sys:
        for X_j in X_env:
            H_SE += J * (X_k @ X_j)

    # Transverse fields on all qubits (system + environment)
    for X_k in X_sys + X_env:
        H_SE += hx * X_k
    for Y_k in Y_sys + Y_env:
        H_SE += hy * Y_k

    return hamiltonian_coupling(H_SE, delta)


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