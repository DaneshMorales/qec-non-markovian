"""
logical_rb — Logical randomized benchmarking under persistent-environment
(non-Markovian) noise.

Quick start
-----------
from logical_rb import RepetitionCode, SteaneCode
from logical_rb import ising_coupling, random_unitary
from logical_rb import run_logical_rb
from logical_rb import fit_rb_curve, non_markovian_diagnostics
from logical_rb import plot_comparison
"""

from .codes import (
    QECCode,
    RepetitionCode,
    SteaneCode,
)

from .noise_models import (
    # Noise model classes
    NoiseModel,
    UnitarySENoise,
    MarkovianKraus,
    TimeVaryingKraus,
    AncillaBitFlipNoise,
    PairwiseCorrelatedNoise,
    StreakCorrelatedNoise,
    PairwisePolyNoise,
    PairwiseExpNoise,
    StreakPolyNoise,
    StreakExpNoise,
    # Decay functions
    poly_decay,
    exp_decay,
    # SE coupling unitary constructors
    hamiltonian_coupling,
    ising_coupling,
    xx_coupling,
    random_unitary,
    partial_swap,
    cnot_env_coupling,
    kraus_from_USE,
)

from .engine import (
    run_logical_rb,
    rb_sequence_survival,
)

from .fitting import (
    rb_model,
    fit_rb_curve,
    non_markovian_diagnostics,
)

from .plotting import (
    plot_comparison,
    plot_markovianization_diagnostics,
    plot_correction_frequency_sweep,
)

from .operators import (
    partial_trace_env,
)

from .cliffords import (
    CLIFFORDS,
    generate_clifford_group,
    clifford_inverse,
    sample_sequence,
    find_recovery_gate,
)


__all__ = [
    # QEC codes
    "QECCode",
    "RepetitionCode",
    "SteaneCode",

    # Noise model classes
    "NoiseModel",
    "UnitarySENoise",
    "MarkovianKraus",
    "TimeVaryingKraus",
    "AncillaBitFlipNoise",
    "PairwiseCorrelatedNoise",
    "StreakCorrelatedNoise",
    "PairwisePolyNoise",
    "PairwiseExpNoise",
    "StreakPolyNoise",
    "StreakExpNoise",
    # Decay functions
    "poly_decay",
    "exp_decay",

    # System-environment noise models
    "hamiltonian_coupling",
    "ising_coupling",
    "xx_coupling",
    "random_unitary",
    "partial_swap",
    "cnot_env_coupling",
    "kraus_from_USE",

    # Logical-RB simulation
    "run_logical_rb",
    "rb_sequence_survival",

    # Curve fitting and diagnostics
    "rb_model",
    "fit_rb_curve",
    "non_markovian_diagnostics",

    # Visualization
    "plot_comparison",
    "plot_markovianization_diagnostics",
    "plot_correction_frequency_sweep",

    # Utility
    "partial_trace_env",

    # Single-qubit Cliffords
    "CLIFFORDS",
    "generate_clifford_group",
    "clifford_inverse",
    "sample_sequence",
    "find_recovery_gate",
]