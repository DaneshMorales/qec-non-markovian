"""
Primitive operators: Pauli matrices, single-qubit gate lifts, and tensor products.

Convention enforced everywhere:
  - Qubit 0 is the leftmost factor in Kronecker products.
  - The full register contains system qubits 0,...,n_sys-1, followed by
    environment qubits n_sys,...,n_sys+n_E-1.
  - A computational-basis state has index

        sum_k q_k * 2**(N - 1 - k),

    where q_k is either 0 or 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# --- Single-qubit Paulis and gates ------------------------------------------

I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0, 1],
     [1, 0]],
    dtype=complex,
)

Y = np.array(
    [[0, -1j],
     [1j, 0]],
    dtype=complex,
)

Z = np.array(
    [[1, 0],
     [0, -1]],
    dtype=complex,
)

H = np.array(
    [[1, 1],
     [1, -1]],
    dtype=complex,
) / np.sqrt(2)

S = np.array(
    [[1, 0],
     [0, 1j]],
    dtype=complex,
)

Sd = S.conj().T


# --- Validation helpers ----------------------------------------------------

def _validate_qubit_count(n: int, name: str, allow_zero: bool = True) -> None:
    """Validate a number of qubits."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"{name} must be an integer, got {type(n).__name__}")

    minimum = 0 if allow_zero else 1

    if n < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {n}")


def _as_2x2_operator(G: np.ndarray) -> np.ndarray:
    """Convert G to a complex 2×2 array and validate its shape."""
    G = np.asarray(G, dtype=complex)

    if G.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 operator, got shape {G.shape}")

    return G


# --- Operator construction -------------------------------------------------

def lift(G: np.ndarray, qubit: int, n_total: int) -> np.ndarray:
    """
    Lift a 2×2 operator G to an n_total-qubit register.

    Qubit 0 is the leftmost Kronecker factor, so the returned operator is

        I^(⊗qubit) ⊗ G ⊗ I^(⊗(n_total-qubit-1)).
    """
    _validate_qubit_count(n_total, "n_total", allow_zero=False)

    if not isinstance(qubit, (int, np.integer)) or isinstance(qubit, bool):
        raise TypeError(
            f"qubit must be an integer, got {type(qubit).__name__}"
        )

    if not 0 <= qubit < n_total:
        raise ValueError(
            f"qubit must satisfy 0 <= qubit < {n_total}, got {qubit}"
        )

    G = _as_2x2_operator(G)

    left_dim = 2**qubit
    right_dim = 2 ** (n_total - qubit - 1)

    left = np.eye(left_dim, dtype=complex)
    right = np.eye(right_dim, dtype=complex)

    return np.kron(np.kron(left, G), right)


def pauli_string(
    ops: Sequence[tuple[np.ndarray, int]],
    n_total: int,
) -> np.ndarray:
    """
    Construct an n_total-qubit tensor-product operator.

    Parameters
    ----------
    ops:
        Sequence of (G, q) pairs, where G is a 2×2 operator acting on
        qubit q. Qubits not included in ops are assigned the identity.

        Each qubit may appear at most once.
    n_total:
        Total number of qubits.

    Returns
    -------
    np.ndarray
        The resulting 2**n_total × 2**n_total operator.
    """
    _validate_qubit_count(n_total, "n_total", allow_zero=False)

    factors = [I2.copy() for _ in range(n_total)]
    occupied_qubits: set[int] = set()

    for G, q in ops:
        if not isinstance(q, (int, np.integer)) or isinstance(q, bool):
            raise TypeError(
                f"Qubit indices must be integers, got {type(q).__name__}"
            )

        if not 0 <= q < n_total:
            raise ValueError(
                f"Qubit index must satisfy 0 <= q < {n_total}, got {q}"
            )

        if q in occupied_qubits:
            raise ValueError(
                f"Qubit {q} appears more than once in the operator string"
            )

        occupied_qubits.add(q)
        factors[q] = _as_2x2_operator(G)

    return kron_list(factors)


def kron_list(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """
    Return the Kronecker product of matrices from left to right.

    The empty Kronecker product is represented by the 1×1 identity.
    """
    if len(matrices) == 0:
        return np.ones((1, 1), dtype=complex)

    out = np.asarray(matrices[0], dtype=complex).copy()

    if out.ndim != 2:
        raise ValueError(
            f"Every element must be a matrix; got shape {out.shape}"
        )

    for M in matrices[1:]:
        M = np.asarray(M, dtype=complex)

        if M.ndim != 2:
            raise ValueError(
                f"Every element must be a matrix; got shape {M.shape}"
            )

        out = np.kron(out, M)

    return out


def lift_to_sys_env(
    op_sys: np.ndarray,
    n_sys: int,
    n_E: int,
) -> np.ndarray:
    """
    Embed a system operator into the full S⊗E Hilbert space.

    Returns

        op_sys ⊗ I_E.
    """
    _validate_qubit_count(n_sys, "n_sys", allow_zero=False)
    _validate_qubit_count(n_E, "n_E", allow_zero=True)

    op_sys = np.asarray(op_sys, dtype=complex)
    expected_shape = (2**n_sys, 2**n_sys)

    if op_sys.shape != expected_shape:
        raise ValueError(
            f"op_sys must have shape {expected_shape}, "
            f"got {op_sys.shape}"
        )

    identity_E = np.eye(2**n_E, dtype=complex)
    return np.kron(op_sys, identity_E)


# --- Environment states ----------------------------------------------------

def env_zero_state(n_E: int) -> np.ndarray:
    """
    Return the environment state |0...0> as a vector of length 2**n_E.

    For n_E = 0, this returns the one-dimensional state [1].
    """
    _validate_qubit_count(n_E, "n_E", allow_zero=True)

    e0 = np.zeros(2**n_E, dtype=complex)
    e0[0] = 1.0

    return e0


def partial_trace_env(
    rho_SE: np.ndarray,
    n_sys: int,
    n_E: int,
) -> np.ndarray:
    """
    Trace out the environment to obtain the system marginal.

    Given a density matrix rho_SE on S⊗E (system qubits leftmost),
    returns

        rho_S = Tr_E[rho_SE].

    Parameters
    ----------
    rho_SE:
        Joint density matrix with shape (2**(n_sys+n_E), 2**(n_sys+n_E)).
    n_sys:
        Number of system qubits.
    n_E:
        Number of environment qubits.

    Returns
    -------
    np.ndarray
        Reduced system density matrix with shape (2**n_sys, 2**n_sys).
    """
    _validate_qubit_count(n_sys, "n_sys", allow_zero=False)
    _validate_qubit_count(n_E, "n_E", allow_zero=True)

    dim_s = 2**n_sys
    dim_e = 2**n_E
    expected_shape = (dim_s * dim_e, dim_s * dim_e)

    rho_SE = np.asarray(rho_SE, dtype=complex)

    if rho_SE.shape != expected_shape:
        raise ValueError(
            f"rho_SE must have shape {expected_shape}, got {rho_SE.shape}"
        )

    # Reshape to (dim_s, dim_e, dim_s, dim_e) then contract over env indices.
    # rho_tensor[i_s, i_e, j_s, j_e] = rho_SE[i_s*dim_e + i_e, j_s*dim_e + j_e]
    # rho_S[i_s, j_s] = sum_{k} rho_tensor[i_s, k, j_s, k]
    rho_tensor = rho_SE.reshape(dim_s, dim_e, dim_s, dim_e)
    return np.einsum('ikjk->ij', rho_tensor)


def reset_env_pure(
    psi: np.ndarray,
    n_sys: int,
    n_E: int,
) -> np.ndarray:
    """
    Approximate an environment reset within pure-state simulation.

    If psi is a product state

        |psi_S> ⊗ |psi_E>,

    this returns, up to a global phase,

        |psi_S> ⊗ |0...0>_E.

    If S and E are entangled, the system marginal is mixed and cannot be
    preserved exactly by a pure state. In that case, this function selects
    the dominant eigenvector of the reduced system density matrix rho_S and
    returns

        |psi_S,dominant> ⊗ |0...0>_E.

    The selected system vector maximizes

        <phi| rho_S |phi>

    over normalized pure system states |phi>.

    For an exact Markovian environment reset of an entangled state, density
    matrices must be used:

        rho_SE -> Tr_E(rho_SE) ⊗ |0><0|_E.
    """
    _validate_qubit_count(n_sys, "n_sys", allow_zero=False)
    _validate_qubit_count(n_E, "n_E", allow_zero=True)

    dim_s = 2**n_sys
    dim_e = 2**n_E
    expected_size = dim_s * dim_e

    psi = np.asarray(psi, dtype=complex)

    if psi.ndim != 1:
        raise ValueError(
            f"psi must be a one-dimensional state vector, got shape {psi.shape}"
        )

    if psi.size != expected_size:
        raise ValueError(
            f"psi must contain {expected_size} amplitudes for "
            f"{n_sys + n_E} qubits, got {psi.size}"
        )

    norm = np.linalg.norm(psi)

    if norm <= 1e-14:
        raise ValueError("psi must be a nonzero state vector")

    # Work with a normalized state.
    psi = psi / norm

    # With system-first ordering, rows index S and columns index E.
    psi_matrix = psi.reshape(dim_s, dim_e)

    # The left singular vectors are eigenvectors of
    # rho_S = psi_matrix @ psi_matrix.conj().T.
    U, singular_values, _ = np.linalg.svd(
        psi_matrix,
        full_matrices=False,
    )

    if singular_values[0] <= 1e-14:
        raise RuntimeError("Failed to obtain a valid system Schmidt vector")

    psi_s_dominant = U[:, 0]
    e0 = env_zero_state(n_E)

    return np.kron(psi_s_dominant, e0)