"""
24-element single-qubit Clifford group: generation, inverse lookup,
and recovery-gate construction.

Uses breadth-first search over the generators {H, S}. Cliffords that differ
only by a global phase are treated as the same group element.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from .operators import I2, H, S


# ---------------------------------------------------------------------------
# Matrix canonicalization
# ---------------------------------------------------------------------------

def _matrix_key(
    U: np.ndarray,
    zero_tol: float = 1e-10,
    decimals: int = 8,
) -> tuple[complex, ...]:
    """
    Return a canonical hashable key for a 2×2 matrix modulo global phase.

    The global phase is fixed by making the first nonzero matrix entry real
    and nonnegative.
    """
    U = np.asarray(U, dtype=complex)

    if U.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 matrix, got shape {U.shape}")

    if not np.all(np.isfinite(U)):
        raise ValueError("Matrix contains nonfinite entries")

    if zero_tol <= 0:
        raise ValueError(f"zero_tol must be positive, got {zero_tol}")

    if (
        not isinstance(decimals, (int, np.integer))
        or isinstance(decimals, (bool, np.bool_))
    ):
        raise TypeError("decimals must be an integer")

    if decimals < 0:
        raise ValueError(f"decimals must be nonnegative, got {decimals}")

    for x in U.ravel():
        if abs(x) > zero_tol:
            # Multiply by exp(-i arg(x)) so that x becomes positive and real.
            phase_correction = abs(x) / x
            canonical = np.round(
                U * phase_correction,
                decimals=int(decimals),
            )

            # Remove harmless signed zeros and tiny numerical residuals.
            canonical.real[
                np.abs(canonical.real) < zero_tol
            ] = 0.0

            canonical.imag[
                np.abs(canonical.imag) < zero_tol
            ] = 0.0

            return tuple(canonical.ravel())

    raise ValueError("Cannot construct a key for the zero matrix")


# ---------------------------------------------------------------------------
# Clifford group generation
# ---------------------------------------------------------------------------

def generate_clifford_group() -> list[np.ndarray]:
    """
    Generate the 24 elements of the single-qubit Clifford group.

    Matrices that differ only by a global phase are identified.
    """
    seen: dict[tuple[complex, ...], np.ndarray] = {}
    queue: deque[np.ndarray] = deque([I2.copy()])

    while queue:
        U = queue.popleft()
        key = _matrix_key(U)

        if key in seen:
            continue

        seen[key] = U

        # Left multiplication by H and S is sufficient to generate the group.
        for generator in (H, S):
            queue.append(generator @ U)

    if len(seen) != 24:
        raise RuntimeError(
            f"Expected 24 single-qubit Cliffords, got {len(seen)}"
        )

    return [U.copy() for U in seen.values()]


CLIFFORDS: list[np.ndarray] = generate_clifford_group()

_KEY_TO_IDX: dict[tuple[complex, ...], int] = {
    _matrix_key(C): i
    for i, C in enumerate(CLIFFORDS)
}


# ---------------------------------------------------------------------------
# Inverse and recovery lookup
# ---------------------------------------------------------------------------

def _distance_from_identity_mod_phase(U: np.ndarray) -> float:
    """
    Return the Frobenius distance from U to the identity modulo global phase.

    Computes

        min_phi ||U - exp(i phi) I||_F.
    """
    U = np.asarray(U, dtype=complex)

    if U.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 matrix, got shape {U.shape}")

    if not np.all(np.isfinite(U)):
        raise ValueError("Matrix contains nonfinite entries")

    overlap = np.trace(U)

    if abs(overlap) <= 1e-12:
        return np.inf

    phase = overlap / abs(overlap)

    return float(
        np.linalg.norm(U - phase * I2, ord="fro")
    )


def clifford_inverse(
    U: np.ndarray,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """
    Return a Clifford recovery C_inv satisfying

        C_inv @ U ≈ exp(i phi) I.

    For an exact Clifford, this is its inverse modulo global phase.

    For a slightly perturbed Clifford, the closest Clifford recovery is
    selected by minimizing the Frobenius distance to the identity modulo
    global phase.

    A ValueError is raised if U is not sufficiently close to a Clifford.
    """
    U = np.asarray(U, dtype=complex)

    if U.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 matrix, got shape {U.shape}")

    if not np.all(np.isfinite(U)):
        raise ValueError("Matrix contains nonfinite entries")

    if not np.isscalar(tolerance) or np.iscomplexobj(tolerance):
        raise TypeError("tolerance must be a real scalar")

    tolerance = float(tolerance)

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            f"tolerance must be finite and positive, got {tolerance}"
        )

    # For an exact unitary Clifford, U^{-1} = U†.
    inverse_key = _matrix_key(U.conj().T)

    if inverse_key in _KEY_TO_IDX:
        index = _KEY_TO_IDX[inverse_key]
        return CLIFFORDS[index].copy()

    # Fallback for small numerical or physical perturbations.
    best_clifford: np.ndarray | None = None
    best_distance = np.inf

    for C in CLIFFORDS:
        distance = _distance_from_identity_mod_phase(C @ U)

        if distance < best_distance:
            best_distance = distance
            best_clifford = C

    if best_clifford is None or best_distance > tolerance:
        raise ValueError(
            "Input is not sufficiently close to a single-qubit Clifford. "
            f"Closest recovery distance was {best_distance:.3e}, "
            f"but the tolerance is {tolerance:.3e}."
        )

    return best_clifford.copy()


def find_recovery_gate(
    sequence: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Return the recovery Clifford for a sequence [C1, C2, ..., Cm].

    The accumulated operation is

        U = Cm @ ... @ C2 @ C1,

    so the returned recovery satisfies

        C_recovery @ U ≈ exp(i phi) I.
    """
    U = I2.copy()

    for index, C in enumerate(sequence):
        C = np.asarray(C, dtype=complex)

        if C.shape != (2, 2):
            raise ValueError(
                f"Sequence element {index} must be 2×2, "
                f"got shape {C.shape}"
            )

        if not np.all(np.isfinite(C)):
            raise ValueError(
                f"Sequence element {index} contains nonfinite entries"
            )

        U = C @ U

    return clifford_inverse(U)


# ---------------------------------------------------------------------------
# Random sequence generation
# ---------------------------------------------------------------------------

def sample_sequence(
    m: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Draw m independent, uniformly random single-qubit Cliffords.

    Copies of the stored Clifford matrices are returned so that modifying a
    sampled sequence does not modify the global CLIFFORDS list.
    """
    if (
        not isinstance(m, (int, np.integer))
        or isinstance(m, (bool, np.bool_))
    ):
        raise TypeError(
            f"m must be a nonnegative integer, got {type(m).__name__}"
        )

    if m < 0:
        raise ValueError(
            f"Sequence length must be nonnegative, got {m}"
        )

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an instance of numpy.random.Generator"
        )

    indices = rng.integers(
        low=0,
        high=len(CLIFFORDS),
        size=int(m),
    )

    return [
        CLIFFORDS[int(index)].copy()
        for index in indices
    ]
