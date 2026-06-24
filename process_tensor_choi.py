"""process_tensor_choi.py

Construct the Choi-state representation of a finite-step process tensor.

The convention follows Appendix A of Figueroa-Romero et al.:

    Upsilon_{k:1} = Tr_E [ (U_k o S_k) ... (U_1 o S_1)
                           (rho_ES tensor psi_{A1B1} tensor ... tensor psi_{AkBk}) ]

where psi = |Omega><Omega| is UNNORMALIZED,
|Omega> = sum_j |j>_A |j>_B, and S_i swaps S with A_i.

Internal subsystem order before tracing E:
    [E, S, A1, B1, A2, B2, ..., Ak, Bk]

Output subsystem order:
    [S, A1, B1, A2, B2, ..., Ak, Bk]

The module contains both:
  * SymPy routines for exact/symbolic calculations.
  * NumPy/SciPy routines for faster numerical calculations.

The full Choi matrix has dimension d_S^(2k+1) by d_S^(2k+1), so the
method is intended for small system dimensions and a modest number of steps.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Sequence

import numpy as np
import sympy as sp
from scipy.linalg import expm


# -----------------------------------------------------------------------------
# Basic tensor utilities
# -----------------------------------------------------------------------------


def kron_all_sympy(matrices: Sequence[sp.MatrixBase]) -> sp.Matrix:
    """Kronecker product of a sequence of SymPy matrices."""
    out = sp.Matrix([[1]])
    for matrix in matrices:
        out = sp.kronecker_product(out, sp.Matrix(matrix))
    return sp.Matrix(out)


def kron_all_numpy(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a sequence of NumPy arrays."""
    out = np.array([[1.0 + 0.0j]])
    for matrix in matrices:
        out = np.kron(out, np.asarray(matrix, dtype=complex))
    return out


def maximally_entangled_operator_sympy(d: int) -> sp.Matrix:
    """Return unnormalized psi = |Omega><Omega| on A tensor B."""
    omega = sp.zeros(d * d, 1)
    for j in range(d):
        omega[j * d + j, 0] = 1
    return omega * omega.H


def maximally_entangled_operator_numpy(d: int) -> np.ndarray:
    """Return unnormalized psi = |Omega><Omega| on A tensor B."""
    omega = np.zeros(d * d, dtype=complex)
    for j in range(d):
        omega[j * d + j] = 1.0
    return np.outer(omega, omega.conj())


def unitary_from_hamiltonian_sympy(
    hamiltonian: sp.MatrixBase,
    time: sp.Expr | float | int = 1,
) -> sp.Matrix:
    """Compute U = exp(-i time H) exactly/symbolically with SymPy."""
    hamiltonian = sp.Matrix(hamiltonian)
    if hamiltonian.rows != hamiltonian.cols:
        raise ValueError("The Hamiltonian must be square.")
    return sp.Matrix((-sp.I * time * hamiltonian).exp())


def unitary_from_hamiltonian_numpy(
    hamiltonian: np.ndarray,
    time: float = 1.0,
) -> np.ndarray:
    """Compute U = exp(-i time H) numerically with SciPy."""
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("The Hamiltonian must be square.")
    return expm(-1.0j * time * hamiltonian)


def _decode_index(index: int, dims: Sequence[int]) -> list[int]:
    """Convert a flat basis index into mixed-radix subsystem indices."""
    digits = [0] * len(dims)
    for position in range(len(dims) - 1, -1, -1):
        digits[position] = index % dims[position]
        index //= dims[position]
    return digits


def _encode_index(digits: Sequence[int], dims: Sequence[int]) -> int:
    """Convert mixed-radix subsystem indices into a flat basis index."""
    index = 0
    for digit, dimension in zip(digits, dims):
        index = index * dimension + digit
    return index


def _swap_inverse_permutation(
    dims: Sequence[int], subsystem_a: int, subsystem_b: int
) -> list[int]:
    """Index list implementing rho -> SWAP rho SWAP^dagger."""
    if dims[subsystem_a] != dims[subsystem_b]:
        raise ValueError("Only equal-dimensional subsystems can be swapped.")

    total_dimension = prod(dims)
    old_to_new = [0] * total_dimension

    for old_index in range(total_dimension):
        digits = _decode_index(old_index, dims)
        digits[subsystem_a], digits[subsystem_b] = (
            digits[subsystem_b],
            digits[subsystem_a],
        )
        old_to_new[old_index] = _encode_index(digits, dims)

    new_to_old = [0] * total_dimension
    for old_index, new_index in enumerate(old_to_new):
        new_to_old[new_index] = old_index
    return new_to_old


def swap_conjugate_sympy(
    operator: sp.MatrixBase,
    dims: Sequence[int],
    subsystem_a: int,
    subsystem_b: int,
) -> sp.Matrix:
    """Conjugate an operator by the SWAP of two equal-dimensional subsystems."""
    inverse_permutation = _swap_inverse_permutation(dims, subsystem_a, subsystem_b)
    return sp.Matrix(operator).extract(inverse_permutation, inverse_permutation)


def swap_conjugate_numpy(
    operator: np.ndarray,
    dims: Sequence[int],
    subsystem_a: int,
    subsystem_b: int,
) -> np.ndarray:
    """Conjugate an operator by the SWAP of two equal-dimensional subsystems."""
    inverse_permutation = _swap_inverse_permutation(dims, subsystem_a, subsystem_b)
    operator = np.asarray(operator, dtype=complex)
    return operator[np.ix_(inverse_permutation, inverse_permutation)]


# -----------------------------------------------------------------------------
# Partial traces
# -----------------------------------------------------------------------------


def partial_trace_sympy(
    operator: sp.MatrixBase,
    dims: Sequence[int],
    trace_out: Sequence[int],
) -> sp.Matrix:
    """Partial trace of a SymPy matrix over the listed subsystem positions.

    This explicit implementation is slow but reliable for the small symbolic
    matrices for which a full process-tensor Choi matrix is practical.
    """
    operator = sp.Matrix(operator)
    dims = list(dims)
    trace_out = sorted(set(trace_out))

    total_dimension = prod(dims)
    if operator.shape != (total_dimension, total_dimension):
        raise ValueError("Operator shape is incompatible with dims.")
    if any(index < 0 or index >= len(dims) for index in trace_out):
        raise ValueError("A traced subsystem index is out of range.")

    keep = [index for index in range(len(dims)) if index not in trace_out]
    keep_dims = [dims[index] for index in keep]
    trace_dims = [dims[index] for index in trace_out]
    output_dimension = prod(keep_dims) if keep_dims else 1
    result = sp.zeros(output_dimension, output_dimension)

    traced_assignments = list(product(*[range(d) for d in trace_dims]))
    if not traced_assignments:
        traced_assignments = [()]

    for output_row in range(output_dimension):
        keep_row_digits = _decode_index(output_row, keep_dims) if keep_dims else []
        for output_col in range(output_dimension):
            keep_col_digits = _decode_index(output_col, keep_dims) if keep_dims else []
            value = sp.S.Zero

            for traced_digits in traced_assignments:
                full_row = [0] * len(dims)
                full_col = [0] * len(dims)

                for position, digit in zip(keep, keep_row_digits):
                    full_row[position] = digit
                for position, digit in zip(keep, keep_col_digits):
                    full_col[position] = digit
                for position, digit in zip(trace_out, traced_digits):
                    full_row[position] = digit
                    full_col[position] = digit

                row_index = _encode_index(full_row, dims)
                col_index = _encode_index(full_col, dims)
                value += operator[row_index, col_index]

            result[output_row, output_col] = value

    return result


def partial_trace_numpy(
    operator: np.ndarray,
    dims: Sequence[int],
    trace_out: Sequence[int],
) -> np.ndarray:
    """Partial trace of a NumPy matrix over the listed subsystem positions."""
    operator = np.asarray(operator, dtype=complex)
    dims = list(dims)
    trace_out = sorted(set(trace_out), reverse=True)

    total_dimension = prod(dims)
    if operator.shape != (total_dimension, total_dimension):
        raise ValueError("Operator shape is incompatible with dims.")
    if any(index < 0 or index >= len(dims) for index in trace_out):
        raise ValueError("A traced subsystem index is out of range.")

    tensor = operator.reshape(tuple(dims + dims))
    for subsystem in trace_out:
        number_of_subsystems = len(dims)
        tensor = np.trace(
            tensor,
            axis1=subsystem,
            axis2=subsystem + number_of_subsystems,
        )
        dims.pop(subsystem)

    remaining_dimension = prod(dims) if dims else 1
    return tensor.reshape((remaining_dimension, remaining_dimension))


# -----------------------------------------------------------------------------
# Process-tensor Choi construction: Appendix A, Eqs. (A3)-(A4)
# -----------------------------------------------------------------------------


def process_tensor_choi_from_unitaries_sympy(
    rho_es: sp.MatrixBase,
    unitaries_es: Sequence[sp.MatrixBase],
    d_e: int,
    d_s: int,
    *,
    simplify_entries: bool = False,
) -> sp.Matrix:
    """Construct Upsilon_{k:1} symbolically from a list U_1,...,U_k.

    Parameters
    ----------
    rho_es:
        Initial density operator in E tensor S ordering.
    unitaries_es:
        Sequence of joint E-S unitaries, each also in E tensor S ordering.
    d_e, d_s:
        Environment and system dimensions.
    simplify_entries:
        Apply sympy.simplify to every final matrix entry. This may be costly.

    Returns
    -------
    SymPy Matrix
        The unnormalized process-tensor Choi state in subsystem order
        [S, A1, B1, ..., Ak, Bk].
    """
    rho_es = sp.Matrix(rho_es)
    unitaries_es = [sp.Matrix(unitary) for unitary in unitaries_es]
    k = len(unitaries_es)

    expected_es_dimension = d_e * d_s
    if rho_es.shape != (expected_es_dimension, expected_es_dimension):
        raise ValueError("rho_es must have shape (d_e*d_s, d_e*d_s).")
    for unitary in unitaries_es:
        if unitary.shape != (expected_es_dimension, expected_es_dimension):
            raise ValueError("Every U_i must have shape (d_e*d_s, d_e*d_s).")

    psi = maximally_entangled_operator_sympy(d_s)
    total_state = kron_all_sympy([rho_es] + [psi] * k)

    # [E, S, A1, B1, ..., Ak, Bk]
    dims = [d_e, d_s] + [d_s, d_s] * k
    auxiliary_dimension = d_s ** (2 * k)
    identity_aux = sp.eye(auxiliary_dimension)

    for step, unitary_es in enumerate(unitaries_es):
        a_i_position = 2 + 2 * step

        # S_i: swap the live system S with A_i.
        total_state = swap_conjugate_sympy(
            total_state, dims, subsystem_a=1, subsystem_b=a_i_position
        )

        # U_i acts on E tensor S; identity acts on all auxiliary systems.
        full_unitary = sp.kronecker_product(unitary_es, identity_aux)
        total_state = full_unitary * total_state * full_unitary.H

    # Trace out E, leaving [S, A1, B1, ..., Ak, Bk].
    upsilon = partial_trace_sympy(total_state, dims, trace_out=[0])

    if simplify_entries:
        upsilon = upsilon.applyfunc(sp.simplify)
    return sp.Matrix(upsilon)


def repeated_unitary_process_tensor_sympy(
    hamiltonian_es: sp.MatrixBase,
    rho_es: sp.MatrixBase,
    k: int,
    d_e: int,
    d_s: int,
    *,
    time: sp.Expr | float | int = 1,
    simplify_entries: bool = False,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Construct a symbolic process tensor for U_i = exp(-i time H) at every step.

    Returns (U, Upsilon_{k:1}).
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    unitary = unitary_from_hamiltonian_sympy(hamiltonian_es, time=time)
    upsilon = process_tensor_choi_from_unitaries_sympy(
        rho_es,
        [unitary] * k,
        d_e,
        d_s,
        simplify_entries=simplify_entries,
    )
    return unitary, upsilon


def process_tensor_choi_from_unitaries_numpy(
    rho_es: np.ndarray,
    unitaries_es: Sequence[np.ndarray],
    d_e: int,
    d_s: int,
) -> np.ndarray:
    """Construct Upsilon_{k:1} numerically from a list U_1,...,U_k."""
    rho_es = np.asarray(rho_es, dtype=complex)
    unitaries_es = [np.asarray(unitary, dtype=complex) for unitary in unitaries_es]
    k = len(unitaries_es)

    expected_es_dimension = d_e * d_s
    if rho_es.shape != (expected_es_dimension, expected_es_dimension):
        raise ValueError("rho_es must have shape (d_e*d_s, d_e*d_s).")
    for unitary in unitaries_es:
        if unitary.shape != (expected_es_dimension, expected_es_dimension):
            raise ValueError("Every U_i must have shape (d_e*d_s, d_e*d_s).")

    psi = maximally_entangled_operator_numpy(d_s)
    total_state = kron_all_numpy([rho_es] + [psi] * k)

    dims = [d_e, d_s] + [d_s, d_s] * k
    auxiliary_dimension = d_s ** (2 * k)
    identity_aux = np.eye(auxiliary_dimension, dtype=complex)

    for step, unitary_es in enumerate(unitaries_es):
        a_i_position = 2 + 2 * step
        total_state = swap_conjugate_numpy(
            total_state, dims, subsystem_a=1, subsystem_b=a_i_position
        )
        full_unitary = np.kron(unitary_es, identity_aux)
        total_state = full_unitary @ total_state @ full_unitary.conj().T

    return partial_trace_numpy(total_state, dims, trace_out=[0])


def repeated_unitary_process_tensor_numpy(
    hamiltonian_es: np.ndarray,
    rho_es: np.ndarray,
    k: int,
    d_e: int,
    d_s: int,
    *,
    time: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a numerical process tensor for U_i = exp(-i time H) at every step.

    Returns (U, Upsilon_{k:1}).
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    unitary = unitary_from_hamiltonian_numpy(hamiltonian_es, time=time)
    upsilon = process_tensor_choi_from_unitaries_numpy(
        rho_es, [unitary] * k, d_e, d_s
    )
    return unitary, upsilon


# -----------------------------------------------------------------------------
# Choi states of interventions and process-tensor contraction: Eqs. (A5)-(A6)
# -----------------------------------------------------------------------------


def choi_from_kraus_sympy(kraus_operators: Sequence[sp.MatrixBase]) -> sp.Matrix:
    """Choi state J_G = (I tensor G)(psi) for a CP map given by Kraus operators."""
    if not kraus_operators:
        raise ValueError("At least one Kraus operator is required.")
    kraus_operators = [sp.Matrix(kraus) for kraus in kraus_operators]
    d = kraus_operators[0].rows
    if any(kraus.shape != (d, d) for kraus in kraus_operators):
        raise ValueError("All Kraus operators must be square and equally sized.")

    psi = maximally_entangled_operator_sympy(d)
    identity = sp.eye(d)
    choi = sp.zeros(d * d, d * d)
    for kraus in kraus_operators:
        embedded = sp.kronecker_product(identity, kraus)
        choi += embedded * psi * embedded.H
    return sp.Matrix(choi)


def choi_from_kraus_numpy(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Choi state J_G = (I tensor G)(psi) for a CP map given by Kraus operators."""
    if not kraus_operators:
        raise ValueError("At least one Kraus operator is required.")
    kraus_operators = [np.asarray(kraus, dtype=complex) for kraus in kraus_operators]
    d = kraus_operators[0].shape[0]
    if any(kraus.shape != (d, d) for kraus in kraus_operators):
        raise ValueError("All Kraus operators must be square and equally sized.")

    psi = maximally_entangled_operator_numpy(d)
    identity = np.eye(d, dtype=complex)
    choi = np.zeros((d * d, d * d), dtype=complex)
    for kraus in kraus_operators:
        embedded = np.kron(identity, kraus)
        choi += embedded @ psi @ embedded.conj().T
    return choi


def contract_process_tensor_sympy(
    upsilon: sp.MatrixBase,
    intervention_kraus: Sequence[Sequence[sp.MatrixBase]],
    d_s: int,
    *,
    simplify_entries: bool = False,
) -> sp.Matrix:
    """Evaluate T_{k:1}[G_1,...,G_k] from Upsilon using Eq. (A5)."""
    k = len(intervention_kraus)
    operation_chois = [choi_from_kraus_sympy(kraus) for kraus in intervention_kraus]
    y_k_to_1 = kron_all_sympy(operation_chois)

    expected_dimension = d_s ** (2 * k + 1)
    upsilon = sp.Matrix(upsilon)
    if upsilon.shape != (expected_dimension, expected_dimension):
        raise ValueError("upsilon has an incompatible dimension.")

    contraction_operator = sp.kronecker_product(sp.eye(d_s), y_k_to_1.T)
    product_operator = upsilon * contraction_operator
    dims = [d_s] + [d_s, d_s] * k
    output = partial_trace_sympy(
        product_operator,
        dims,
        trace_out=list(range(1, len(dims))),
    )
    if simplify_entries:
        output = output.applyfunc(sp.simplify)
    return sp.Matrix(output)


def contract_process_tensor_numpy(
    upsilon: np.ndarray,
    intervention_kraus: Sequence[Sequence[np.ndarray]],
    d_s: int,
) -> np.ndarray:
    """Evaluate T_{k:1}[G_1,...,G_k] from Upsilon using Eq. (A5)."""
    k = len(intervention_kraus)
    operation_chois = [choi_from_kraus_numpy(kraus) for kraus in intervention_kraus]
    y_k_to_1 = kron_all_numpy(operation_chois)

    expected_dimension = d_s ** (2 * k + 1)
    upsilon = np.asarray(upsilon, dtype=complex)
    if upsilon.shape != (expected_dimension, expected_dimension):
        raise ValueError("upsilon has an incompatible dimension.")

    contraction_operator = np.kron(np.eye(d_s, dtype=complex), y_k_to_1.T)
    product_operator = upsilon @ contraction_operator
    dims = [d_s] + [d_s, d_s] * k
    return partial_trace_numpy(
        product_operator,
        dims,
        trace_out=list(range(1, len(dims))),
    )


# -----------------------------------------------------------------------------
# Direct evolution and diagnostics
# -----------------------------------------------------------------------------


def direct_process_output_numpy(
    rho_es: np.ndarray,
    unitaries_es: Sequence[np.ndarray],
    intervention_kraus: Sequence[Sequence[np.ndarray]],
    d_e: int,
    d_s: int,
) -> np.ndarray:
    """Directly simulate G_1 -> U_1 -> ... -> G_k -> U_k and trace E."""
    if len(unitaries_es) != len(intervention_kraus):
        raise ValueError("There must be one intervention for each unitary step.")

    rho = np.asarray(rho_es, dtype=complex)
    identity_e = np.eye(d_e, dtype=complex)

    for unitary, kraus_operators in zip(unitaries_es, intervention_kraus):
        after_operation = np.zeros_like(rho, dtype=complex)
        for kraus in kraus_operators:
            embedded = np.kron(identity_e, np.asarray(kraus, dtype=complex))
            after_operation += embedded @ rho @ embedded.conj().T
        unitary = np.asarray(unitary, dtype=complex)
        rho = unitary @ after_operation @ unitary.conj().T

    return partial_trace_numpy(rho, [d_e, d_s], trace_out=[0])


def process_tensor_diagnostics_numpy(
    upsilon: np.ndarray,
    d_s: int,
    k: int,
    *,
    atol: float = 1e-10,
) -> dict[str, float | bool]:
    """Return basic checks: Hermiticity, positivity, and trace d_S^k."""
    upsilon = np.asarray(upsilon, dtype=complex)
    hermiticity_error = float(np.linalg.norm(upsilon - upsilon.conj().T))
    hermitian_part = 0.5 * (upsilon + upsilon.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_part)).real)
    actual_trace = complex(np.trace(upsilon))
    expected_trace = float(d_s**k)

    return {
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "trace_real": float(actual_trace.real),
        "trace_imag_abs": float(abs(actual_trace.imag)),
        "expected_trace": expected_trace,
        "is_hermitian": hermiticity_error <= atol,
        "is_positive_semidefinite": minimum_eigenvalue >= -atol,
        "has_expected_trace": abs(actual_trace - expected_trace) <= atol,
    }


# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------


def symbolic_ising_example() -> tuple[sp.Matrix, sp.Matrix]:
    """Two-qubit E-S Ising example with symbolic J and tau, using k=2."""
    J, tau = sp.symbols("J tau", real=True)
    z = sp.diag(1, -1)

    # Convention: H acts on E tensor S.
    h_es = J * sp.kronecker_product(z, z)

    # rho_ES = |00><00|, also in E tensor S ordering.
    ket_00 = sp.Matrix([1, 0, 0, 0])
    rho_es = ket_00 * ket_00.H

    unitary, upsilon = repeated_unitary_process_tensor_sympy(
        h_es,
        rho_es,
        k=2,
        d_e=2,
        d_s=2,
        time=tau,
        simplify_entries=True,
    )
    return unitary, upsilon


def numerical_consistency_example() -> float:
    """Compare Choi contraction against direct dynamics for a two-step example."""
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    coupling = 0.3
    local_field = 0.2
    h_es = coupling * np.kron(z, z) + local_field * np.kron(identity, x)
    unitary = unitary_from_hamiltonian_numpy(h_es)

    ket_00 = np.array([1, 0, 0, 0], dtype=complex)
    rho_es = np.outer(ket_00, ket_00.conj())

    upsilon = process_tensor_choi_from_unitaries_numpy(
        rho_es,
        [unitary, unitary],
        d_e=2,
        d_s=2,
    )

    g_1 = expm(-1.0j * 0.4 * x)
    g_2 = expm(-1.0j * 0.2 * z)
    interventions = [[g_1], [g_2]]

    output_from_choi = contract_process_tensor_numpy(
        upsilon, interventions, d_s=2
    )
    output_direct = direct_process_output_numpy(
        rho_es,
        [unitary, unitary],
        interventions,
        d_e=2,
        d_s=2,
    )

    return float(np.max(np.abs(output_from_choi - output_direct)))


if __name__ == "__main__":
    U_symbolic, Upsilon_symbolic = symbolic_ising_example()
    print("Symbolic repeated unitary U =")
    sp.pprint(U_symbolic)
    print("\nUpsilon shape:", Upsilon_symbolic.shape)
    print("Tr(Upsilon) =", sp.simplify(sp.trace(Upsilon_symbolic)))

    numerical_error = numerical_consistency_example()
    print("\nMaximum Choi-vs-direct numerical error:", numerical_error)


# =============================================================================
# QEC-aware process-tensor construction (noise_models.py integration)
# =============================================================================


# -----------------------------------------------------------------------------
# Subsystem reordering: noise_models uses S⊗E; process tensor uses E⊗S
# -----------------------------------------------------------------------------


def reorder_se_to_es(U_se: np.ndarray, d_s: int, d_e: int) -> np.ndarray:
    """Convert a unitary from S⊗E tensor ordering to E⊗S ordering.

    ``noise_models.two_spin_coupling`` places system qubits first and the
    environment qubit last, giving a matrix in the S⊗E basis.  The
    process-tensor routines in this module expect E⊗S ordering (environment
    first).  This function performs the index permutation
    [s_row, e_row, s_col, e_col] → [e_row, s_row, e_col, s_col].
    """
    U_se = np.asarray(U_se, dtype=complex)
    T = U_se.reshape(d_s, d_e, d_s, d_e)
    T = T.transpose(1, 0, 3, 2)
    return T.reshape(d_s * d_e, d_s * d_e)


def reorder_es_to_se(U_es: np.ndarray, d_s: int, d_e: int) -> np.ndarray:
    """Inverse of ``reorder_se_to_es``: E⊗S → S⊗E."""
    return reorder_se_to_es(U_es, d_e, d_s)


def initial_rho_es(n_sys: int, n_E: int) -> np.ndarray:
    """Return ρ_ES = |0⟩⟨0|_E ⊗ |0…0⟩⟨0…0|_S in E⊗S ordering.

    This is a pure product state with environment and system both in |0⟩.
    """
    dim = 2 ** (n_sys + n_E)
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    return rho


# -----------------------------------------------------------------------------
# QEC recovery map as Kraus operators
# -----------------------------------------------------------------------------


def qec_recovery_kraus_numpy(code) -> list[np.ndarray]:
    """Build Kraus operators for the full QEC syndrome-measure-and-correct channel.

    For a stabilizer code with generators {g_j} and a correction table
    syndrome → R_s, the Kraus operators are

        K_s = R_s @ P_s,

    where P_s = ∏_j (I + (−1)^{s_j} g_j) / 2 is the projector onto the
    syndrome-s eigenspace.  The operators satisfy ∑_s K_s† K_s = I exactly.

    Parameters
    ----------
    code:
        Any QECCode object exposing ``.stabilizers`` (list of n-qubit matrices)
        and ``.recovery(syndrome)`` (returns the correction unitary for the
        given syndrome tuple).

    Returns
    -------
    list of (2^n × 2^n) complex numpy arrays.
    """
    from itertools import product as iproduct

    n = code.n
    dim = 2 ** n
    I_n = np.eye(dim, dtype=complex)
    n_stab = len(code.stabilizers)

    kraus_ops: list[np.ndarray] = []
    for syndrome in iproduct([0, 1], repeat=n_stab):
        # Syndrome projector P_s = ∏_j (I ± g_j) / 2
        P_s = I_n.copy()
        for j, g_j in enumerate(code.stabilizers):
            sign = (-1) ** syndrome[j]
            P_s = P_s @ (I_n + sign * g_j) / 2

        # Skip zero projectors (numerically)
        if np.trace(P_s).real < 1e-10:
            continue

        R_s = code.recovery(syndrome)
        K_s = R_s @ P_s
        kraus_ops.append(K_s)

    return kraus_ops


def identity_kraus_numpy(d: int) -> list[np.ndarray]:
    """Single-element Kraus list [I_d] representing the identity channel."""
    return [np.eye(d, dtype=complex)]


# -----------------------------------------------------------------------------
# Process-tensor construction with optional QEC recovery
# -----------------------------------------------------------------------------


def process_tensor_choi_with_qec_numpy(
    rho_es: np.ndarray,
    unitaries_es: Sequence[np.ndarray],
    d_e: int,
    d_s: int,
    recovery_kraus: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Process-tensor Choi state with an optional QEC recovery after each step.

    At every time step i the evolution is

        1. SWAP(S, A_i)                — records pre-noise system state
        2. Apply U_i on E⊗S            — SE noise unitary
        3. Apply R on S (identity on E) — QEC recovery (or identity if None)

    Setting ``recovery_kraus=None`` gives the *bare* (no-QEC) process tensor,
    equivalent to ``process_tensor_choi_from_unitaries_numpy``.

    Parameters
    ----------
    rho_es:
        Initial E⊗S density matrix, shape (d_e*d_s, d_e*d_s).
    unitaries_es:
        k unitaries in E⊗S ordering, each shape (d_e*d_s, d_e*d_s).
    d_e, d_s:
        Environment and system dimensions.
    recovery_kraus:
        Kraus operators for the QEC recovery map on S only, each (d_s, d_s).
        If None, the identity channel is used (no QEC).

    Returns
    -------
    Upsilon in [S, A1, B1, ..., Ak, Bk] subsystem ordering,
    shape (d_s^{2k+1}, d_s^{2k+1}).
    """
    rho_es = np.asarray(rho_es, dtype=complex)
    unitaries_es = [np.asarray(U, dtype=complex) for U in unitaries_es]
    k = len(unitaries_es)

    if recovery_kraus is None:
        recovery_kraus = identity_kraus_numpy(d_s)
    recovery_kraus = [np.asarray(K, dtype=complex) for K in recovery_kraus]

    expected_es_dim = d_e * d_s
    if rho_es.shape != (expected_es_dim, expected_es_dim):
        raise ValueError("rho_es must have shape (d_e*d_s, d_e*d_s).")
    for U in unitaries_es:
        if U.shape != (expected_es_dim, expected_es_dim):
            raise ValueError("Every U_i must have shape (d_e*d_s, d_e*d_s).")
    for K in recovery_kraus:
        if K.shape != (d_s, d_s):
            raise ValueError("Every recovery Kraus op must have shape (d_s, d_s).")

    # Build initial state: ρ_ES ⊗ |Ω⟩⟨Ω|^⊗k
    psi_anc = maximally_entangled_operator_numpy(d_s)
    total_state = kron_all_numpy([rho_es] + [psi_anc] * k)

    # Subsystem dims: [E, S, A1, B1, ..., Ak, Bk]
    dims = [d_e, d_s] + [d_s, d_s] * k
    aux_dim = d_s ** (2 * k)
    I_e = np.eye(d_e, dtype=complex)
    I_aux = np.eye(aux_dim, dtype=complex)

    for step, U_es in enumerate(unitaries_es):
        a_i = 2 + 2 * step

        # ── Step 1: SWAP(S, A_i) ─────────────────────────────────────────────
        total_state = swap_conjugate_numpy(total_state, dims, 1, a_i)

        # ── Step 2: U_i on E⊗S, identity on aux ─────────────────────────────
        U_full = np.kron(U_es, I_aux)
        total_state = U_full @ total_state @ U_full.conj().T

        # ── Step 3: Recovery R on S (position 1), identity on E and aux ──────
        if len(recovery_kraus) == 1 and np.allclose(recovery_kraus[0], np.eye(d_s)):
            pass  # identity — skip
        else:
            new_state = np.zeros_like(total_state)
            for K in recovery_kraus:
                # Embed K into full space: I_E ⊗ K_S ⊗ I_aux
                K_full = np.kron(np.kron(I_e, K), I_aux)
                new_state += K_full @ total_state @ K_full.conj().T
            total_state = new_state

    # Trace out E (subsystem 0)
    return partial_trace_numpy(total_state, dims, trace_out=[0])


# -----------------------------------------------------------------------------
# Entanglement entropy and information-theoretic diagnostics
# -----------------------------------------------------------------------------


def von_neumann_entropy(rho: np.ndarray, *, atol: float = 1e-12) -> float:
    """Von Neumann entropy S(ρ) = −Tr[ρ log ρ] (base-2 logarithm).

    Non-positive eigenvalues smaller than ``atol`` are treated as zero.
    """
    rho = np.asarray(rho, dtype=complex)
    eigs = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real
    eigs = eigs[eigs > atol]
    return float(-np.sum(eigs * np.log2(eigs)))


def entanglement_entropy_bipartition(
    upsilon: np.ndarray,
    dims: Sequence[int],
    subsystems_A: Sequence[int],
) -> float:
    """Entanglement entropy of Upsilon/Tr(Upsilon) across a subsystem bipartition.

    Parameters
    ----------
    upsilon:
        Process-tensor Choi matrix (not necessarily normalised).
    dims:
        List of subsystem dimensions consistent with upsilon.
    subsystems_A:
        Indices of the subsystems in partition A; the rest form partition B.

    Returns
    -------
    Von Neumann entropy S(ρ_A) in bits (base-2 log).
    """
    upsilon = np.asarray(upsilon, dtype=complex)
    tr = np.trace(upsilon).real
    if tr < 1e-14:
        raise ValueError("Process tensor has zero trace.")
    rho = upsilon / tr

    subsystems_B = [i for i in range(len(dims)) if i not in subsystems_A]
    rho_A = partial_trace_numpy(rho, list(dims), subsystems_B)
    return von_neumann_entropy(rho_A)


def temporal_mutual_information(
    upsilon: np.ndarray,
    d_s: int,
    k: int,
    cut: int | None = None,
) -> float:
    """Temporal mutual information I(A:B) across a bipartition of the process tensor.

    The process tensor subsystems are ordered [S, A1, B1, ..., Ak, Bk].
    This function traces out S, then computes

        I(A:B) = S(ρ_A) + S(ρ_B) − S(ρ_AB)

    where A = steps 1..cut and B = steps cut+1..k (each 'step' is the pair
    {A_i, B_i}).

    I(A:B) = 0 iff the two halves are uncorrelated, i.e. ρ_AB = ρ_A ⊗ ρ_B.
    For Markovian noise (U acts on S only, no SE coupling) the process tensor
    factorises across time steps and I = 0 exactly.  For non-Markovian SE
    coupling I > 0 because the environment carries memory between steps.

    Parameters
    ----------
    cut:
        Number of steps in partition A.  Defaults to k // 2.
    """
    if cut is None:
        cut = k // 2
    if cut < 1 or cut >= k:
        raise ValueError(f"cut must be in [1, k-1], got {cut} with k={k}.")

    # Trace out S_out (position 0); normalise
    dims_full = [d_s] + [d_s, d_s] * k
    rho_AB = partial_trace_numpy(upsilon, dims_full, trace_out=[0])
    dims_anc = [d_s, d_s] * k          # [A1, B1, ..., Ak, Bk]
    tr = np.trace(rho_AB).real
    rho_AB = rho_AB / tr

    S_AB = von_neumann_entropy(rho_AB)

    # Partition A = steps 1..cut  (indices 0 .. 2*cut-1 in dims_anc)
    # Partition B = steps cut+1..k (indices 2*cut .. 2*k-1)
    rho_A = partial_trace_numpy(rho_AB, dims_anc, list(range(2 * cut, 2 * k)))
    rho_B = partial_trace_numpy(rho_AB, dims_anc, list(range(0, 2 * cut)))

    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)

    return max(0.0, S_A + S_B - S_AB)   # max(0,·) absorbs numerical noise


def temporal_entanglement_entropy(
    upsilon: np.ndarray,
    d_s: int,
    k: int,
    cut: int | None = None,
) -> float:
    """Alias for temporal_mutual_information (kept for backwards compat)."""
    return temporal_mutual_information(upsilon, d_s, k, cut)


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Trace distance D(ρ1, ρ2) = ½ Tr|ρ1 − ρ2|."""
    diff = np.asarray(rho1, dtype=complex) - np.asarray(rho2, dtype=complex)
    eigvals = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigvals)))


def process_non_markovianity(upsilon: np.ndarray, upsilon_markov_ref: np.ndarray) -> float:
    """Trace distance between a process tensor and its Markovian reference.

    The Markovian reference Υ_reset is computed with ``env_reset=True`` (the
    environment is reset to |0_E⟩ after each noise step, removing all memory).

    D(Υ/Tr Υ, Υ_reset/Tr Υ_reset) = 0  iff the process is Markovian (no
    environment coupling, or coupling that leaves E unchanged).  For SE
    coupling without QEC it is > 0; QEC reduces it by projecting the system
    back into the code space after each step, partially breaking the chain of
    environment memory.
    """
    rho1 = upsilon / np.trace(upsilon).real
    rho2 = upsilon_markov_ref / np.trace(upsilon_markov_ref).real
    return trace_distance(rho1, rho2)


# -----------------------------------------------------------------------------
# High-level comparison: bare vs QEC process tensor
# -----------------------------------------------------------------------------


def compare_qec_process_tensors(
    code,
    U_se_raw: np.ndarray,
    k: int,
    n_E: int = 1,
    *,
    se_ordering: str = "SE",
    verbose: bool = True,
) -> dict:
    """Compare the process tensors with and without QEC for a given SE unitary.

    This is the main entry point for process-tensor analysis.  It:

    1. Converts ``U_se_raw`` from S⊗E ordering (noise_models convention) to
       E⊗S ordering (process-tensor convention) unless ``se_ordering='ES'``.
    2. Initialises the environment + system in |0⟩.
    3. Builds Upsilon_bare  (no recovery, R = identity at each step).
    4. Builds Upsilon_qec   (QEC recovery map applied after every U_SE step).
    5. Computes and returns information-theoretic diagnostics.

    Parameters
    ----------
    code:
        A QECCode object (RepetitionCode, FiveQubitCode, etc.).
    U_se_raw:
        The SE unitary from ``two_spin_coupling`` or similar, in S⊗E ordering.
    k:
        Number of noise + intervention steps.
    n_E:
        Number of environment qubits.
    se_ordering:
        ``'SE'`` (default, noise_models convention) or ``'ES'`` (already E⊗S).
    verbose:
        Print a formatted summary.

    Returns
    -------
    dict with keys:
        ``upsilon_bare``, ``upsilon_qec``  — the two Choi matrices.
        ``entropy_bare``, ``entropy_qec``   — von Neumann entropy (bits).
        ``temporal_ee_bare``, ``temporal_ee_qec``  — temporal EE (k ≥ 2).
        ``trace_distance``                  — D(bare, qec) process tensors.
        ``diagnostics_bare``, ``diagnostics_qec`` — hermiticity/trace checks.
    """
    d_s = 2 ** code.n
    d_e = 2 ** n_E

    # Memory guard: physical Upsilon has d_s^{2k+1} × d_s^{2k+1} elements × 16 B
    upsilon_dim = d_s ** (2 * k + 1)
    mem_gb = 16 * upsilon_dim ** 2 / 1e9
    if upsilon_dim > 4096:
        raise MemoryError(
            f"Physical process tensor for {code.__class__.__name__} (n={code.n}, d_s={d_s}) "
            f"with k={k} would be {upsilon_dim}×{upsilon_dim} ({mem_gb:.1f} GB). "
            f"Use compare_qec_process_tensors_logical() instead."
        )

    # ── 1.  Ordering conversion ───────────────────────────────────────────────
    U_se_raw = np.asarray(U_se_raw, dtype=complex)
    if se_ordering == "SE":
        U_es = reorder_se_to_es(U_se_raw, d_s, d_e)
    else:
        U_es = U_se_raw

    unitaries = [U_es] * k

    # ── 2.  Initial state |0⟩_E ⊗ |0…0⟩_S ─────────────────────────────────
    rho_init = initial_rho_es(code.n, n_E)  # in E⊗S ordering

    # ── 3.  No-QEC process tensor ─────────────────────────────────────────────
    upsilon_bare = process_tensor_choi_with_qec_numpy(
        rho_init, unitaries, d_e, d_s, recovery_kraus=None
    )

    # ── 4.  QEC process tensor ────────────────────────────────────────────────
    R_kraus = qec_recovery_kraus_numpy(code)
    upsilon_qec = process_tensor_choi_with_qec_numpy(
        rho_init, unitaries, d_e, d_s, recovery_kraus=R_kraus
    )

    # ── 5.  Normalize both ───────────────────────────────────────────────────
    tr_bare = np.trace(upsilon_bare).real
    tr_qec  = np.trace(upsilon_qec).real
    rho_bare = upsilon_bare / tr_bare
    rho_qec  = upsilon_qec  / tr_qec

    # ── 6.  Diagnostics ───────────────────────────────────────────────────────
    diag_bare = process_tensor_diagnostics_numpy(upsilon_bare, d_s, k)
    diag_qec  = process_tensor_diagnostics_numpy(upsilon_qec,  d_s, k)

    S_bare = von_neumann_entropy(rho_bare)
    S_qec  = von_neumann_entropy(rho_qec)
    D      = trace_distance(rho_bare, rho_qec)

    tee_bare = temporal_mutual_information(upsilon_bare, d_s, k) if k >= 2 else float("nan")
    tee_qec  = temporal_mutual_information(upsilon_qec,  d_s, k) if k >= 2 else float("nan")

    if verbose:
        w = 35
        print(f"\n{'Process-tensor comparison':{'='}^{2*w+3}}")
        print(f"  Code:    {code.__class__.__name__}  (n={code.n}, d_S={d_s})")
        print(f"  n_E={n_E}, k={k} steps")
        print(f"  Upsilon size: {upsilon_bare.shape[0]} × {upsilon_bare.shape[1]}")
        print()
        header = f"  {'Quantity':<{w}}  {'Bare (no QEC)':>12}  {'With QEC':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        rows = [
            ("Von Neumann entropy S (bits)",      S_bare,    S_qec),
            ("Temporal MI I(A:B) (cut=k//2, bits)", tee_bare, tee_qec),
            ("Trace distance D(bare, qec)",        D,         None),
            ("Tr(Upsilon) / d_S^k",
             tr_bare / d_s**k, tr_qec / d_s**k),
            ("Min eigenvalue",
             diag_bare["minimum_eigenvalue"], diag_qec["minimum_eigenvalue"]),
        ]
        for name, v_bare, v_qec in rows:
            v_b = f"{v_bare:12.6f}" if v_bare is not None else "           —"
            v_q = (f"{v_qec:12.6f}" if v_qec is not None else "           —")
            print(f"  {name:<{w}}  {v_b}  {v_q}")
        print()

    return {
        "upsilon_bare":      upsilon_bare,
        "upsilon_qec":       upsilon_qec,
        "entropy_bare":      S_bare,
        "entropy_qec":       S_qec,
        "temporal_ee_bare":  tee_bare,
        "temporal_ee_qec":   tee_qec,
        "trace_distance":    D,
        "diagnostics_bare":  diag_bare,
        "diagnostics_qec":   diag_qec,
    }


# =============================================================================
# Logical-level process tensor (scales with k, not d_s^k)
# =============================================================================


def _apply_matrix_on_subsystems(
    rho: np.ndarray,
    dims: list,
    positions: list,
    M: np.ndarray,
) -> np.ndarray:
    """Return M ρ M† with M acting on the listed subsystem positions.

    M must be square with side length equal to the product of dims at
    the given positions.  For a Kraus channel, call this once per
    operator and sum the results.

    Uses two BLAS matmuls (M @ T_flat and M.conj() @ T_rearranged) so
    performance scales as O(d_sub³ × D_passive²) rather than O(d_sub⁶).
    """
    n = len(dims)
    D = int(np.prod(dims))
    passive = [i for i in range(n) if i not in positions]
    d_sub = int(np.prod([dims[p] for p in positions]))
    D_passive = D // d_sub

    # Permute so active positions come first in both ket and bra
    perm = list(positions) + passive
    perm2 = perm + [p + n for p in perm]

    # Bring to (d_sub, D_passive, d_sub, D_passive) — copy once for contiguity
    T = np.ascontiguousarray(rho.reshape(list(dims) * 2).transpose(perm2))
    T = T.reshape(d_sub, D_passive, d_sub, D_passive)

    # Step 1: Y[a,r,d,s] = Σ_c M[a,c] T[c,r,d,s]  (BLAS matmul)
    Y = (M @ T.reshape(d_sub, -1)).reshape(d_sub, D_passive, d_sub, D_passive)

    # Step 2: Z[a,r,b,s] = Σ_d Y[a,r,d,s] M*[b,d]  (BLAS matmul)
    Yt = np.ascontiguousarray(Y.transpose(2, 0, 1, 3)).reshape(d_sub, -1)
    T = (M.conj() @ Yt).reshape(d_sub, d_sub, D_passive, D_passive).transpose(1, 2, 0, 3)

    all_dims = [dims[perm[i]] for i in range(n)]
    T = T.reshape(all_dims * 2)

    inv_perm2 = [0] * (2 * n)
    for j, p in enumerate(perm2):
        inv_perm2[p] = j
    return np.ascontiguousarray(T.transpose(inv_perm2)).reshape(D, D)


def _logical_vswap(d_s: int, U_enc: np.ndarray) -> np.ndarray:
    """Build the logical-SWAP unitary V on C^{d_s} ⊗ C^2.

    Maps  |i_L⟩_S |j⟩_A  →  |j_L⟩_S |i⟩_A  for i,j ∈ {0,1},
    and leaves the non-code subspace of S invariant.

    V = (U_enc ⊗ I₂) SWAP₂₂ (U_enc† ⊗ I₂) + (P_⊥ ⊗ I₂)
    where SWAP₂₂ is the 4×4 SWAP gate and P_⊥ = I − U_enc U_enc†.
    """
    P_perp = np.eye(d_s, dtype=complex) - U_enc @ U_enc.conj().T
    SWAP_22 = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
    )
    A = np.kron(U_enc, np.eye(2, dtype=complex))  # (2*d_s, 4)
    return A @ SWAP_22 @ A.conj().T + np.kron(P_perp, np.eye(2, dtype=complex))


def logical_process_tensor_numpy(
    code,
    U_se_raw: np.ndarray,
    k: int,
    n_E: int = 1,
    *,
    apply_qec: bool = True,
    se_ordering: str = "SE",
    env_reset: bool = False,
) -> np.ndarray:
    """k-step process tensor at the LOGICAL qubit level.

    Output subsystem ordering: [L_out, A1, B1, …, Ak, Bk], each subsystem
    2-dimensional.  Output shape: (2^{2k+1}, 2^{2k+1}).

    The intermediate computation propagates the state in the physical+
    environment space of dimension  d_e × d_s × 4^k,  which is far smaller
    than d_s^{2k+1} for multi-qubit codes.  Even with k=4, a RepetitionCode
    (n=3) uses a 512×512 intermediate matrix.

    Parameters
    ----------
    code:
        Must expose ``encode_zero()``, ``encode_one()``, and (when
        apply_qec=True) ``stabilizers`` / ``recovery()``.
    apply_qec:
        Apply the QEC recovery channel after each noise step (default True).
        Set to False for the "bare" no-QEC process tensor.
    env_reset:
        If True, reset E to |0_E⟩⟨0_E| after each noise step before applying
        QEC.  This removes all environment memory between steps and produces
        the *Markovian reference* process tensor.  Compare Υ to this reference
        to quantify non-Markovianity: D(Υ/Tr, Υ_reset/Tr) = 0 iff the
        process is Markovian (i.e. U = I_E ⊗ U_S).
    """
    d_s = 2 ** code.n
    d_e = 2 ** n_E

    U_se_raw = np.asarray(U_se_raw, dtype=complex)
    U_es = reorder_se_to_es(U_se_raw, d_s, d_e) if se_ordering == "SE" else U_se_raw.copy()

    # Encoding isometry: columns = |0_L⟩, |1_L⟩
    ez = np.asarray(code.encode_zero(), dtype=complex)  # (d_s,)
    eo = np.asarray(code.encode_one(),  dtype=complex)  # (d_s,)
    U_enc = np.column_stack([ez, eo])                    # (d_s, 2)

    V_swap = _logical_vswap(d_s, U_enc)  # (2*d_s, 2*d_s) unitary

    R_kraus = qec_recovery_kraus_numpy(code) if apply_qec else identity_kraus_numpy(d_s)
    is_id_R = len(R_kraus) == 1 and np.allclose(R_kraus[0], np.eye(d_s))

    # Initial environment state (reused for reset)
    rho_E_init = np.zeros((d_e, d_e), dtype=complex); rho_E_init[0, 0] = 1.0

    # Initial state: ρ_E = |0⟩⟨0|, system = |0_L⟩⟨0_L| (in E⊗S ordering)
    rho_0L = np.outer(ez, ez.conj())
    total_state = np.kron(rho_E_init, rho_0L)

    # Logical ancilla pairs |Ω_L⟩⟨Ω_L|, |Ω_L⟩ = |00⟩ + |11⟩ (unnormalised)
    psi_L = np.zeros((4, 4), dtype=complex)
    psi_L[0, 0] = psi_L[0, 3] = psi_L[3, 0] = psi_L[3, 3] = 1.0

    for _ in range(k):
        total_state = np.kron(total_state, psi_L)

    # Subsystem dims: [E(d_e), S(d_s), A1(2), B1(2), …, Ak(2), Bk(2)]
    dims = [d_e, d_s] + [2, 2] * k

    for step in range(k):
        a_pos = 2 + 2 * step  # position of A_{step+1}

        # Step 1: logical SWAP on (S=pos1, A_i=a_pos)
        total_state = _apply_matrix_on_subsystems(total_state, dims, [1, a_pos], V_swap)

        # Step 2: U_ES on (E=0, S=1)
        total_state = _apply_matrix_on_subsystems(total_state, dims, [0, 1], U_es)

        # Optional: reset E to |0_E><0_E| to break environment memory
        if env_reset:
            dims_no_e = dims[1:]
            rho_no_e = partial_trace_numpy(total_state, dims, trace_out=[0])
            total_state = np.kron(rho_E_init, rho_no_e)

        # Step 3: QEC recovery (Kraus channel) on S=1
        if not is_id_R:
            new_state = np.zeros_like(total_state)
            for K in R_kraus:
                new_state += _apply_matrix_on_subsystems(total_state, dims, [1], K)
            total_state = new_state

    # Trace out E (position 0)
    upsilon_phys = partial_trace_numpy(total_state, dims, trace_out=[0])
    # dims now: [S(d_s), A1(2), B1(2), …, Ak(2), Bk(2)]

    # Decode S: contract with U_enc† (isometry d_s → 2)
    D_rest = 4 ** k
    T = upsilon_phys.reshape(d_s, D_rest, d_s, D_rest)
    # T_new[i,r,j,s] = Σ_{a,b} U_enc*[a,i] T[a,r,b,s] U_enc[b,j]
    upsilon_L = np.einsum("ai,arbs,bj->irjs", U_enc.conj(), T, U_enc)
    return upsilon_L.reshape(2 * D_rest, 2 * D_rest)


def compare_qec_process_tensors_logical(
    code,
    U_se_raw: np.ndarray,
    k: int,
    n_E: int = 1,
    *,
    se_ordering: str = "SE",
    verbose: bool = True,
) -> dict:
    """Compare bare vs QEC process tensors at the LOGICAL qubit level.

    Builds the k-step logical process tensor (dim 2^{2k+1} × 2^{2k+1}) for
    both the bare (no-QEC) and QEC cases, then reports information-theoretic
    diagnostics.

    Parameters
    ----------
    code:
        QECCode with ``encode_zero()``, ``encode_one()``, ``stabilizers``,
        ``recovery()``.
    U_se_raw:
        SE unitary in S⊗E ordering (noise_models convention).
    k:
        Number of noise steps.

    Returns
    -------
    dict with keys:
        ``upsilon_bare``, ``upsilon_qec``       — logical process tensors.
        ``entropy_bare``, ``entropy_qec``        — S(Υ/Tr Υ) in bits.
        ``temporal_ee_bare``, ``temporal_ee_qec``— temporal EE (k ≥ 2).
        ``trace_distance``                       — D(bare, qec).
        ``diagnostics_bare``, ``diagnostics_qec``— trace/PSD checks.
    """
    d_L = 2  # logical qubit dimension

    upsilon_bare = logical_process_tensor_numpy(
        code, U_se_raw, k, n_E, apply_qec=False, se_ordering=se_ordering
    )
    upsilon_qec = logical_process_tensor_numpy(
        code, U_se_raw, k, n_E, apply_qec=True, se_ordering=se_ordering
    )

    # Markovian references: same dynamics but environment reset after each step
    upsilon_bare_ref = logical_process_tensor_numpy(
        code, U_se_raw, k, n_E, apply_qec=False, se_ordering=se_ordering, env_reset=True
    )
    upsilon_qec_ref = logical_process_tensor_numpy(
        code, U_se_raw, k, n_E, apply_qec=True, se_ordering=se_ordering, env_reset=True
    )

    tr_bare = np.trace(upsilon_bare).real
    tr_qec  = np.trace(upsilon_qec).real
    rho_bare = upsilon_bare / tr_bare
    rho_qec  = upsilon_qec  / tr_qec

    S_bare = von_neumann_entropy(rho_bare)
    S_qec  = von_neumann_entropy(rho_qec)
    D_dist = trace_distance(rho_bare, rho_qec)

    # Non-Markovianity: distance from the corresponding Markovian reference
    # N = 0 for Markovian noise (U = I_E ⊗ U_S), N > 0 for SE coupling
    N_bare = process_non_markovianity(upsilon_bare, upsilon_bare_ref)
    N_qec  = process_non_markovianity(upsilon_qec,  upsilon_qec_ref)

    diag_bare = process_tensor_diagnostics_numpy(upsilon_bare, d_L, k)
    diag_qec  = process_tensor_diagnostics_numpy(upsilon_qec,  d_L, k)

    if verbose:
        w = 38
        dim = 2 ** (2 * k + 1)
        print(f"\n{'Logical process-tensor comparison':{'='}^{2*w+3}}")
        print(f"  Code: {code.__class__.__name__}  (n={code.n}, d_S=2^{code.n}={2**code.n})")
        print(f"  n_E={n_E}, k={k} steps  →  logical Υ size {dim}×{dim}")
        print()
        header = f"  {'Quantity':<{w}}  {'Bare (no QEC)':>12}  {'With QEC':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        rows = [
            ("Von Neumann entropy S (bits)",             S_bare,   S_qec),
            ("Non-Markovianity D(Υ, Υ_reset)",           N_bare,   N_qec),
            ("Trace distance D(bare, qec)",               D_dist,   None),
            ("Tr(Υ) / d_L^k = Tr(Υ) / 2^k",
             tr_bare / d_L ** k, tr_qec / d_L ** k),
            ("Min eigenvalue",
             diag_bare["minimum_eigenvalue"], diag_qec["minimum_eigenvalue"]),
        ]
        for name, v_b, v_q in rows:
            s_b = f"{v_b:12.6f}" if v_b is not None else "           —"
            s_q = (f"{v_q:12.6f}" if v_q is not None else "           —")
            print(f"  {name:<{w}}  {s_b}  {s_q}")
        print()

    return {
        "upsilon_bare":          upsilon_bare,
        "upsilon_qec":           upsilon_qec,
        "upsilon_bare_ref":      upsilon_bare_ref,
        "upsilon_qec_ref":       upsilon_qec_ref,
        "entropy_bare":          S_bare,
        "entropy_qec":           S_qec,
        "non_markovianity_bare": N_bare,
        "non_markovianity_qec":  N_qec,
        "trace_distance":        D_dist,
        "diagnostics_bare":      diag_bare,
        "diagnostics_qec":       diag_qec,
    }
