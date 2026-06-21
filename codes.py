"""
QEC codes: three-qubit repetition code and Steane [[7,1,3]] code.

Each code exposes:
    n
        Number of physical qubits.

    stabilizers
        Stabilizer generators in the same order expected by recovery().

    x_stabilizers
        X-type stabilizer generators.

    z_stabilizers
        Z-type stabilizer generators.

    logical_x, logical_z
        Physical representatives of the logical Pauli operators.

    logical_zero_projector
        Rank-one projector |0_L><0_L|.

    logical_z_plus_projector
        Full-space projector (I + Z_L)/2.

    code_projector
        Projector onto the two-dimensional code space.

    encode_zero(), encode_one()
        Encoded logical basis states.

    encoder()
        Encoding isometry V whose columns are |0_L> and |1_L>.

    logical_unitary(C)
        Extension of a 2x2 logical unitary to the physical Hilbert space.

    recovery(syndrome)
        Pauli correction associated with a measured syndrome.

Convention:
    Qubit 0 is the leftmost Kronecker factor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .operators import I2, X, Z, H, Sd, kron_list, pauli_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_syndrome(
    syndrome: Sequence[int],
    expected_length: int,
    code_name: str,
) -> tuple[int, ...]:
    """Validate a binary syndrome and return it as a tuple of integers."""
    try:
        bits = tuple(syndrome)
    except TypeError as exc:
        raise TypeError(
            f"{code_name} syndrome must be a sequence of bits"
        ) from exc

    if len(bits) != expected_length:
        raise ValueError(
            f"{code_name} syndrome must contain exactly "
            f"{expected_length} bits, got {len(bits)}"
        )

    validated: list[int] = []

    for index, bit in enumerate(bits):
        if (
            not isinstance(bit, (int, np.integer, bool, np.bool_))
            or int(bit) not in (0, 1)
        ):
            raise ValueError(
                f"{code_name} syndrome entry {index} must be 0 or 1, "
                f"got {bit!r}"
            )

        validated.append(int(bit))

    return tuple(validated)


def _state_projector(psi: np.ndarray) -> np.ndarray:
    """Return the rank-one projector |psi><psi|."""
    psi = np.asarray(psi, dtype=complex)

    if psi.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional state vector, got shape {psi.shape}"
        )

    norm = np.linalg.norm(psi)

    if norm <= 1e-14:
        raise ValueError("Cannot construct a projector from the zero vector")

    psi = psi / norm
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QECCode(ABC):
    """Abstract base class for a quantum error-correcting code."""

    n: int

    stabilizers: list[np.ndarray]
    x_stabilizers: list[np.ndarray]
    z_stabilizers: list[np.ndarray]

    logical_x: np.ndarray
    logical_z: np.ndarray

    logical_zero_projector: np.ndarray
    logical_z_plus_projector: np.ndarray
    code_projector: np.ndarray

    @abstractmethod
    def encode_zero(self) -> np.ndarray:
        """Return the encoded logical-zero state."""

    @abstractmethod
    def encode_one(self) -> np.ndarray:
        """Return the encoded logical-one state."""

    def encoder(self) -> np.ndarray:
        """
        Return the encoding isometry V with

            V[:, 0] = |0_L>,
            V[:, 1] = |1_L>.
        """
        zero = np.asarray(self.encode_zero(), dtype=complex)
        one = np.asarray(self.encode_one(), dtype=complex)

        expected_shape = (2**self.n,)

        if zero.shape != expected_shape:
            raise ValueError(
                f"encode_zero() must return shape {expected_shape}, "
                f"got {zero.shape}"
            )

        if one.shape != expected_shape:
            raise ValueError(
                f"encode_one() must return shape {expected_shape}, "
                f"got {one.shape}"
            )

        V = np.column_stack((zero, one))

        if not np.allclose(
            V.conj().T @ V,
            I2,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Encoded logical basis states are not orthonormal"
            )

        return V

    def logical_unitary(self, C: np.ndarray) -> np.ndarray:
        """
        Extend a 2x2 logical unitary C to the full physical Hilbert space.

        The extension is

            U = V C V† + (I - V V†),

        so U acts as C on the code space and as identity on its orthogonal
        complement.
        """
        C = np.asarray(C, dtype=complex)

        if C.shape != (2, 2):
            raise ValueError(
                f"C must be a 2x2 matrix, got shape {C.shape}"
            )

        if not np.all(np.isfinite(C)):
            raise ValueError("C contains nonfinite entries")

        if not np.allclose(
            C.conj().T @ C,
            I2,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("C must be unitary")

        V = self.encoder()
        P = V @ V.conj().T
        identity = np.eye(2**self.n, dtype=complex)

        return V @ C @ V.conj().T + identity - P

    @abstractmethod
    def recovery(
        self,
        syndrome: Sequence[int],
    ) -> np.ndarray:
        """Return the correction operator associated with a syndrome."""


# ---------------------------------------------------------------------------
# Three-qubit repetition code
# ---------------------------------------------------------------------------

class RepetitionCode(QECCode):
    """
    Three-qubit repetition code for correcting one X error.

    Stabilizers:
        Z0 Z1,
        Z1 Z2.

    Logical operators:
        X_L = X0 X1 X2,
        Z_L = Z0.

    Syndrome table:
        (0, 0) -> I,
        (1, 0) -> X0,
        (1, 1) -> X1,
        (0, 1) -> X2.

    This code has restricted bit-flip distance 3 but full quantum distance 1.
    It does not protect against arbitrary single-qubit Pauli errors.
    """

    n = 3

    def __init__(self) -> None:
        dim = 2**self.n
        identity = np.eye(dim, dtype=complex)

        Z0Z1 = kron_list((Z, Z, I2))
        Z1Z2 = kron_list((I2, Z, Z))

        self.z_stabilizers = [Z0Z1, Z1Z2]
        self.x_stabilizers = []

        # This ordering matches recovery().
        self.stabilizers = self.z_stabilizers.copy()

        self.logical_x = kron_list((X, X, X))
        self.logical_z = kron_list((Z, I2, I2))

        zero = self.encode_zero()
        one = self.encode_one()

        self.logical_zero_projector = _state_projector(zero)

        # This is not a rank-one projector on the full physical space.
        self.logical_z_plus_projector = (
            identity + self.logical_z
        ) / 2

        self.code_projector = (
            _state_projector(zero)
            + _state_projector(one)
        )

        X0 = kron_list((X, I2, I2))
        X1 = kron_list((I2, X, I2))
        X2 = kron_list((I2, I2, X))

        self._recovery_table: dict[
            tuple[int, int],
            np.ndarray,
        ] = {
            (0, 0): identity,
            (1, 0): X0,
            (1, 1): X1,
            (0, 1): X2,
        }

    def encode_zero(self) -> np.ndarray:
        """Return |0_L> = |000>."""
        psi = np.zeros(8, dtype=complex)
        psi[0] = 1.0
        return psi

    def encode_one(self) -> np.ndarray:
        """Return |1_L> = |111>."""
        psi = np.zeros(8, dtype=complex)
        psi[7] = 1.0
        return psi

    def recovery(
        self,
        syndrome: Sequence[int],
    ) -> np.ndarray:
        """Return the X correction associated with a two-bit syndrome."""
        key = _validate_syndrome(
            syndrome,
            expected_length=2,
            code_name="Repetition-code",
        )

        return self._recovery_table[key].copy()


# ---------------------------------------------------------------------------
# Steane [[7,1,3]] code
# ---------------------------------------------------------------------------

class SteaneCode(QECCode):
    """
    Steane [[7,1,3]] CSS code.

    X stabilizers, which detect Z errors:
        g_X0 = X0 X2 X4 X6,
        g_X1 = X1 X2 X5 X6,
        g_X2 = X3 X4 X5 X6.

    Z stabilizers, which detect X errors:
        g_Z0 = Z0 Z2 Z4 Z6,
        g_Z1 = Z1 Z2 Z5 Z6,
        g_Z2 = Z3 Z4 Z5 Z6.

    Logical operators:
        X_L = X^tensor 7,
        Z_L = Z^tensor 7.

    Stabilizer and syndrome ordering:
        stabilizers = z_stabilizers + x_stabilizers

    Therefore recovery() expects

        (z_s0, z_s1, z_s2, x_s0, x_s1, x_s2).

    For three syndrome bits,

        index = s0 + 2*s1 + 4*s2.

    Index zero means no error. Index k in {1,...,7} means the error occurred
    on zero-indexed physical qubit k-1.
    """

    n = 7

    _X_SUPPORTS: tuple[tuple[int, ...], ...] = (
        (0, 2, 4, 6),
        (1, 2, 5, 6),
        (3, 4, 5, 6),
    )

    _Z_SUPPORTS = _X_SUPPORTS

    def __init__(self) -> None:
        dim = 2**self.n
        identity = np.eye(dim, dtype=complex)

        self.x_stabilizers = [
            pauli_string(
                [(X, qubit) for qubit in support],
                self.n,
            )
            for support in self._X_SUPPORTS
        ]

        self.z_stabilizers = [
            pauli_string(
                [(Z, qubit) for qubit in support],
                self.n,
            )
            for support in self._Z_SUPPORTS
        ]

        # This order matches the syndrome convention in recovery().
        self.stabilizers = (
            self.z_stabilizers
            + self.x_stabilizers
        )

        self.logical_x = kron_list((X,) * self.n)
        self.logical_z = kron_list((Z,) * self.n)

        self._zero = self._project_basis_state_to_code(0)
        self._one = self.logical_x @ self._zero
        self._one /= np.linalg.norm(self._one)

        self._validate_codewords()

        self.logical_zero_projector = _state_projector(
            self._zero
        )

        # Full-space logical-Z +1 eigenspace projector.
        self.logical_z_plus_projector = (
            identity + self.logical_z
        ) / 2

        self.code_projector = (
            _state_projector(self._zero)
            + _state_projector(self._one)
        )

        self.transversal_H = kron_list((H,) * self.n)
        self.transversal_Sd = kron_list((Sd,) * self.n)

        self._x_corrections: dict[int, np.ndarray] = {
            0: identity,
            **{
                index: pauli_string(
                    [(X, index - 1)],
                    self.n,
                )
                for index in range(1, 8)
            },
        }

        self._z_corrections: dict[int, np.ndarray] = {
            0: identity,
            **{
                index: pauli_string(
                    [(Z, index - 1)],
                    self.n,
                )
                for index in range(1, 8)
            },
        }

    def _project_basis_state_to_code(
        self,
        basis_index: int,
    ) -> np.ndarray:
        """
        Project a computational-basis state into the simultaneous +1
        eigenspace of all stabilizer generators.
        """
        dim = 2**self.n

        if (
            not isinstance(basis_index, (int, np.integer))
            or isinstance(basis_index, (bool, np.bool_))
        ):
            raise TypeError("basis_index must be an integer")

        if not 0 <= basis_index < dim:
            raise ValueError(
                f"basis_index must satisfy 0 <= basis_index < {dim}, "
                f"got {basis_index}"
            )

        psi = np.zeros(dim, dtype=complex)
        psi[int(basis_index)] = 1.0

        for stabilizer in self.stabilizers:
            psi = (psi + stabilizer @ psi) / 2

        norm = np.linalg.norm(psi)

        if norm <= 1e-12:
            raise RuntimeError(
                "The chosen basis state has zero projection onto "
                "the Steane code space"
            )

        return psi / norm

    def _validate_codewords(self) -> None:
        """Verify the encoded basis states and logical-Z eigenvalues."""
        if not np.isclose(
            np.linalg.norm(self._zero),
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError("Logical zero is not normalized")

        if not np.isclose(
            np.linalg.norm(self._one),
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError("Logical one is not normalized")

        if not np.isclose(
            np.vdot(self._zero, self._one),
            0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Logical zero and logical one are not orthogonal"
            )

        for index, stabilizer in enumerate(self.stabilizers):
            if not np.allclose(
                stabilizer @ self._zero,
                self._zero,
                atol=1e-10,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"Logical zero is not stabilized by generator {index}"
                )

            if not np.allclose(
                stabilizer @ self._one,
                self._one,
                atol=1e-10,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"Logical one is not stabilized by generator {index}"
                )

        if not np.allclose(
            self.logical_z @ self._zero,
            self._zero,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Logical zero does not have logical-Z eigenvalue +1"
            )

        if not np.allclose(
            self.logical_z @ self._one,
            -self._one,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Logical one does not have logical-Z eigenvalue -1"
            )

    def encode_zero(self) -> np.ndarray:
        """Return the Steane logical-zero state."""
        return self._zero.copy()

    def encode_one(self) -> np.ndarray:
        """Return the Steane logical-one state."""
        return self._one.copy()

    def recovery(
        self,
        syndrome: Sequence[int],
    ) -> np.ndarray:
        """
        Return the CSS Pauli correction for a six-bit syndrome.

        Syndrome ordering:

            (z_s0, z_s1, z_s2, x_s0, x_s1, x_s2).

        Z-stabilizer bits detect X errors.
        X-stabilizer bits detect Z errors.
        """
        bits = _validate_syndrome(
            syndrome,
            expected_length=6,
            code_name="Steane-code",
        )

        z_syndrome = bits[:3]
        x_syndrome = bits[3:]

        x_index = (
            z_syndrome[0]
            + 2 * z_syndrome[1]
            + 4 * z_syndrome[2]
        )

        z_index = (
            x_syndrome[0]
            + 2 * x_syndrome[1]
            + 4 * x_syndrome[2]
        )

        recovery_operator = (
            self._x_corrections[x_index]
            @ self._z_corrections[z_index]
        )

        return recovery_operator.copy()
