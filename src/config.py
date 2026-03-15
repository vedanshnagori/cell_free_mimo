"""
Configuration for Cell-Free MIMO QNN System
"""

import numpy as np


class NetworkConfig:
    """Configuration for the wireless network simulation"""

    # Network topology — NAP > Nuser required (Section III)
    NUM_ACCESS_POINTS = 6       # NAP, must be > NUM_USERS
    NUM_USERS         = 3       # Nuser, Table III: Nuser = 3
    NUM_ANTENNAS      = 2       # NTx,  Table III: NTx = 2

    # Channel parameters (Section V, Table III)
    CARRIER_FREQUENCY    = 2.4e9   # 2.4 GHz
    BANDWIDTH            = 20e6    # 20 MHz
    PATH_LOSS_EXPONENT   = 2.3     # κ = 2.3, Table III
    NUM_PATHS            = 10      # Npath for channel model Eq. (3)

    # Power parameters for SNR ρ = Pt/σ² (Eq. 4)
    TRANSMIT_POWER_DBM   = 20
    NOISE_POWER_DBM      = -90
    NOISE_POWER_LINEAR   = 10 ** (NOISE_POWER_DBM  / 10) * 1e-3
    TRANSMIT_POWER_LINEAR= 10 ** (TRANSMIT_POWER_DBM / 10) * 1e-3
    SNR                  = TRANSMIT_POWER_LINEAR / NOISE_POWER_LINEAR

    # Interference factor μ_n,k from Eq. (4), Table III
    INTERFERENCE_FACTOR  = 0.1

    # Area dimensions (in meters)
    AREA_WIDTH           = 200
    AREA_HEIGHT          = 200

    # Training parameters (Algorithm 1, Section V, Table III)
    N_DATA               = 100
    N_EPOCH              = 100
    NUM_ITERATIONS_CLOUD = 100
    NUM_ITERATIONS_EDGE  = 100
    LEARNING_RATE        = 0.01    # μ, Table III

    # Penalty for unassigned users Eq. (14), Table III
    R_PENALTY            = -10.0

    # Simulation parameters (Section V)
    NUM_CHANNEL_REALIZATIONS = 10
    NUM_MONTE_CARLO_TRIALS   = 1000
    RANDOM_SEED              = 42


class QNNConfig:
    """Configuration for Quantum Neural Network"""

    # Circuit parameters (Lemma 1)
    NUM_QUBITS_CLOUD = 6        # for U^cloud_connect
    NUM_QUBITS_EDGE  = 3        # = Nuser (Lemma 1: edge needs Nuser qubits)
    NUM_LAYERS       = 3        # N^cloud_layer
    ENTANGLEMENT     = 'linear' # matches paper's sequential CZ gates
    REPS             = 2        # repetitions in ansatz

    # Optimizer settings (Appendix B)
    OPTIMIZER        = 'COBYLA'
    MAX_ITER         = 100
    USE_ROTOSOLVE    = True     # Appendix B alternative optimizer
    USE_PARAM_SHIFT  = True     # parameter shift rule, Appendix B

    # Descending learning rate (Section V)
    # μ_itrain = μ / √(i_episode + 1)
    LEARNING_RATE    = 0.01
    LR_DECAY         = True

    # Backend settings
    BACKEND          = 'qasm_simulator'
    SHOTS            = 8192

    # Feature map
    FEATURE_MAP      = 'ZZFeatureMap'

    # ── Shared parameters (also needed by CloudQNN and EdgeQNN) ────
    # Reproducibility
    RANDOM_SEED          = 42       # matches NetworkConfig.RANDOM_SEED

    # SNR ρ = Pt/σ² (Eq. 4)
    TRANSMIT_POWER_DBM   = 20
    NOISE_POWER_DBM      = -90
    NOISE_POWER_LINEAR   = 10 ** (NOISE_POWER_DBM  / 10) * 1e-3
    TRANSMIT_POWER_LINEAR= 10 ** (TRANSMIT_POWER_DBM / 10) * 1e-3
    SNR                  = TRANSMIT_POWER_LINEAR / NOISE_POWER_LINEAR

    # Interference factor μ_n,k (Eq. 4), Table III
    INTERFERENCE_FACTOR  = 0.1

    # Penalty for unassigned users (Eq. 14), Table III
    R_PENALTY            = -10.0

    # Training iterations
    NUM_ITERATIONS_CLOUD = 100
    NUM_ITERATIONS_EDGE  = 100