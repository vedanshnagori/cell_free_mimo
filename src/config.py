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
    TRANSMIT_POWER_DBM   = 20                                    # Pt
    NOISE_POWER_DBM      = -90                                   # σ² in dBm
    NOISE_POWER_LINEAR   = 10 ** (NOISE_POWER_DBM / 10) * 1e-3  # σ² in Watts
    TRANSMIT_POWER_LINEAR= 10 ** (TRANSMIT_POWER_DBM / 10)*1e-3 # Pt in Watts
    SNR                  = TRANSMIT_POWER_LINEAR/NOISE_POWER_LINEAR # ρ

    # Interference factor μ_n,k from Eq. (4)
    # Table III: μ_n,k = 0.1
    INTERFERENCE_FACTOR  = 0.1     # μ_n,k

    # Area dimensions (in meters)
    AREA_WIDTH           = 200
    AREA_HEIGHT          = 200

    # Training parameters (Algorithm 1, Section V, Table III)
    N_DATA               = 100     # Ndata,  Table III
    N_EPOCH              = 100     # Nepoch, Table III
    NUM_ITERATIONS_CLOUD = 50      # cloud training iterations
    NUM_ITERATIONS_EDGE  = 30      # edge training iterations
    LEARNING_RATE        = 0.01    # μ, Table III

    # Penalty for unassigned users Eq. (14)
    R_PENALTY            = -10.0   # r_penalty, Table III

    # Simulation parameters (Section V)
    NUM_CHANNEL_REALIZATIONS = 10
    NUM_MONTE_CARLO_TRIALS   = 1000  # 10³ trials in paper
    RANDOM_SEED              = 42


class QNNConfig:
    """Configuration for Quantum Neural Network"""

    # Circuit parameters (Lemma 1)
    NUM_QUBITS_CLOUD = 8        # for U^cloud_connect
    NUM_QUBITS_EDGE  = 3        # = Nuser (Lemma 1: edge needs Nuser qubits)
    NUM_LAYERS       = 3        # N^cloud_layer
    ENTANGLEMENT     = 'linear' # matches paper's sequential CZ gates
    REPS             = 2        # repetitions in ansatz

    # Optimizer settings (Appendix B)
    OPTIMIZER        = 'COBYLA'  # or 'SPSA', 'ADAM'
    MAX_ITER         = 100
    USE_ROTOSOLVE    = True      # Appendix B alternative optimizer
    USE_PARAM_SHIFT  = True      # parameter shift rule, Appendix B

    # Descending learning rate (Section V)
    # μ_itrain = μ / √(i_episode + 1)
    LEARNING_RATE    = 0.01      # base μ
    LR_DECAY         = True      # enable μ/√(episode+1) schedule

    # Backend settings
    BACKEND          = 'qasm_simulator'
    SHOTS            = 4096      # increased from 1024 for stability

    # Feature map
    FEATURE_MAP      = 'ZZFeatureMap'
