"""
Noise model classes and SE coupling unitary constructors.

Noise model classes (pass to run_logical_rb / rb_sequence_survival)
--------------------------------------------------------------------
UnitarySENoise(U_SE, n_E)
    Non-Markovian: persistent environment coupled via a fixed unitary U_SE.
    Supports Markovian reference (reset_E=True) via exact partial-trace reset.

MarkovianKraus(kraus_ops)
    Markovian CPTP channel, same Kraus operators every cycle, no environment.

TimeVaryingKraus(kraus_ops_list)
    Time-structured CPTP channel without a quantum environment: the Kraus
    operators change deterministically per gate cycle (periodic).

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
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .operators import I2, X, Y, Z, lift


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