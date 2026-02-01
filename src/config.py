"""
Configuration file for QNN-based Wireless Network Optimization
Contains all simulation parameters and network settings
"""

import numpy as np

class NetworkConfig:
    """Configuration for the wireless network simulation"""
    
    # Network topology
    NUM_ACCESS_POINTS = 4  # Number of APs (base stations)
    NUM_USERS = 6  # Number of users
    NUM_ANTENNAS = 4  # Number of antennas per AP
    
    # Channel parameters
    CARRIER_FREQUENCY = 2.4e9  # 2.4 GHz
    BANDWIDTH = 20e6  # 20 MHz
    NOISE_POWER_DBM = -90  # Noise power in dBm
    PATH_LOSS_EXPONENT = 3.5
    
    # Area dimensions (in meters)
    AREA_WIDTH = 200
    AREA_HEIGHT = 200
    
    # QNN parameters
    NUM_QUBITS_CLOUD = 8  # Number of qubits for cloud QNN
    NUM_QUBITS_EDGE = 6   # Number of qubits for edge QNN
    NUM_LAYERS = 3        # Number of QNN layers
    
    # Training parameters
    NUM_ITERATIONS_CLOUD = 50  # Training iterations for cloud QNN
    NUM_ITERATIONS_EDGE = 30   # Training iterations for edge QNN
    LEARNING_RATE = 0.01
    
    # Simulation parameters
    NUM_CHANNEL_REALIZATIONS = 10
    RANDOM_SEED = 42

class QNNConfig:
    """Configuration for Quantum Neural Network"""
    
    # Circuit parameters
    ENTANGLEMENT = 'linear'  # or 'full', 'circular'
    REPS = 2  # Number of repetitions in ansatz
    
    # Optimizer settings
    OPTIMIZER = 'COBYLA'  # or 'SPSA', 'ADAM'
    MAX_ITER = 100
    
    # Backend settings
    BACKEND = 'qasm_simulator'  # Qiskit Aer simulator
    SHOTS = 1024
    
    # Feature map
    FEATURE_MAP = 'ZZFeatureMap'  # or 'ZFeatureMap', 'PauliFeatureMap'
    
