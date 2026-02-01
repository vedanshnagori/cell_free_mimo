# Multi-Stage QNN for Wireless Network Optimization

This project implements a **Multi-Stage Quantum Neural Network (QNN)** system for wireless network optimization, based on the research paper's algorithms. The system uses quantum computing to solve the joint transmitter-user assignment and transmit precoding optimization problem in cell-free networks.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Algorithm Implementation](#algorithm-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [File Structure](#file-structure)
- [Technical Details](#technical-details)
- [Citation](#citation)

## 🎯 Overview

This simulation implements three main algorithms from the paper:

1. **Algorithm 1: Multi-Stage QNNs** - Main orchestration of cloud and edge QNNs
2. **Algorithm 2: Transmitter-User Assignment** - Cloud QNN for optimal user-AP assignment
3. **Algorithm 3: Transmit Precoding Optimization** - Edge QNNs for precoding vector calculation

### Key Features

- ✅ Full implementation of multi-stage QNN architecture
- ✅ Cloud QNN for centralized assignment decisions
- ✅ Distributed Edge QNNs for local precoding optimization
- ✅ Realistic wireless channel modeling (Rayleigh fading + path loss)
- ✅ Quantum circuit implementation using Qiskit
- ✅ Training and deployment phases
- ✅ Performance visualization and metrics

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLOUD QNN                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │  Input: Complete Channel Information H            │     │
│  │  Output: Assignment Policy γ                      │     │
│  │  QNN: 8 qubits, ZZ Feature Map, Real Amplitudes  │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (broadcasts γ)
┌─────────────────────────────────────────────────────────────┐
│                    EDGE QNNs (Per AP)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Edge 1     │  │   Edge 2     │  │   Edge m     │     │
│  │ Input: h₁, γ │  │ Input: h₂, γ │  │ Input: hₘ, γ │     │
│  │ Output: v₁   │  │ Output: v₂   │  │ Output: vₘ   │     │
│  │ 6 qubits     │  │ 6 qubits     │  │ 6 qubits     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                   Network Deployment
```

## 🔧 Algorithm Implementation

### Algorithm 1: Multi-Stage QNNs

**Phases:**

1. **Initialization**: Define APs and users
2. **Cloud QNN Training**: Train with complete channel information
3. **Edge QNN Training**: Each AP trains locally with assignment policy
4. **Deployment**: Real-time assignment and precoding

**Implementation**: `src/multi_stage_qnn.py`

### Algorithm 2: Transmitter-User Assignment (Cloud QNN)

**Steps:**

1. Initialize qubits in |0⟩ state
2. For each training iteration:
   - Encode channel information using feature map (Eq. 10, 11)
   - Apply variational quantum circuit
   - Decode to assignment policy γ (Eq. 9)
   - Calculate quality Q_assign (Eq. 14)
   - Compute loss L_assign (Eq. 13)
   - Update parameters using gradient descent

**Implementation**: `src/cloud_qnn.py`

### Algorithm 3: Transmit Precoding Optimization (Edge QNN)

**Steps:**

1. Initialize qubits for each AP
2. For each training iteration:
   - Encode local channel hₘ and assignment γ (Eq. 19)
   - Apply parameterized quantum circuit (Eq. 20)
   - Decode to precoding vectors vₘ
   - Calculate quality Q_precode (Eq. 16)
   - Compute loss L_precode (Eq. 17)
   - Update parameters

**Implementation**: `src/edge_qnn.py`

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project:
```bash
cd qnn_wireless_optimization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- **qiskit**: Quantum computing framework
- **qiskit-aer**: High-performance quantum simulator
- **qiskit-algorithms**: Quantum algorithms and optimizers
- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **scipy**: Scientific computing

## 🚀 Usage

### Basic Usage

Run the complete simulation:

```bash
python main.py
```

### Expected Output

The simulation will:

1. Initialize network topology (APs and users)
2. Train Cloud QNN for user assignment
3. Train Edge QNNs for precoding
4. Deploy and evaluate performance
5. Save results and visualizations

### Output Files

Results are saved in the `results/` directory:

- `results_YYYYMMDD_HHMMSS.json`: Numerical results and metrics
- `visualization_YYYYMMDD_HHMMSS.png`: Network topology and performance plots

## ⚙️ Configuration

### Network Parameters

Edit `src/config.py` to modify simulation parameters:

```python
class NetworkConfig:
    NUM_ACCESS_POINTS = 4      # Number of APs
    NUM_USERS = 6              # Number of users
    NUM_ANTENNAS = 4           # Antennas per AP
    AREA_WIDTH = 200           # Network area (meters)
    AREA_HEIGHT = 200
    
    NUM_ITERATIONS_CLOUD = 50  # Cloud training iterations
    NUM_ITERATIONS_EDGE = 30   # Edge training iterations
    LEARNING_RATE = 0.01
```

### QNN Parameters

```python
class QNNConfig:
    NUM_QUBITS_CLOUD = 8       # Qubits for cloud QNN
    NUM_QUBITS_EDGE = 6        # Qubits per edge QNN
    ENTANGLEMENT = 'linear'    # 'linear', 'full', or 'circular'
    REPS = 2                   # Circuit repetitions
    OPTIMIZER = 'COBYLA'       # Quantum optimizer
    SHOTS = 1024               # Measurement shots
```

## 📊 Results

### Performance Metrics

The simulation reports:

- **Sum Rate**: Total achievable rate (bits/s/Hz)
- **Average SINR**: Signal-to-interference-plus-noise ratio (dB)
- **Active Users**: Number of successfully assigned users
- **Assignment Matrix**: User-to-AP mapping
- **Training Convergence**: Loss curves for cloud and edge QNNs

### Visualization

The generated plots show:

1. **Network Topology**: AP positions, user positions, and assignment connections
2. **Cloud QNN Training**: Loss convergence over iterations
3. **Assignment Heatmap**: Visual representation of user-AP assignments
4. **Performance Metrics**: Bar chart of key performance indicators

## 📁 File Structure

```
qnn_wireless_optimization/
│
├── main.py                    # Main simulation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── src/                       # Source code
│   ├── config.py             # Configuration parameters
│   ├── channel_model.py      # Wireless channel generation
│   ├── cloud_qnn.py          # Cloud QNN (Algorithm 2)
│   ├── edge_qnn.py           # Edge QNN (Algorithm 3)
│   └── multi_stage_qnn.py    # Multi-stage system (Algorithm 1)
│
├── data/                      # Data files (generated)
├── tests/                     # Unit tests
├── results/                   # Simulation results (generated)
└── docs/                      # Additional documentation
```

## 🔬 Technical Details

### Quantum Circuit Architecture

#### Cloud QNN
- **Encoding**: ZZ Feature Map with linear entanglement
- **Ansatz**: Real Amplitudes with parameterized rotation gates
- **Qubits**: 8 (configurable)
- **Layers**: 3 repetitions
- **Measurements**: Computational basis

#### Edge QNN
- **Encoding**: ZZ Feature Map for local channel information
- **Ansatz**: Real Amplitudes for precoding optimization
- **Qubits**: 6 per edge (configurable)
- **Layers**: 2 repetitions
- **Measurements**: Computational basis

### Channel Model

The wireless channel is modeled with:

- **Small-scale fading**: Rayleigh distribution (complex Gaussian)
- **Large-scale fading**: Path loss with exponent 3.5
- **Path loss equation**: PL(d) = PL₀ + 10n·log₁₀(d/d₀)
- **Frequency**: 2.4 GHz
- **Bandwidth**: 20 MHz

### Optimization Process

1. **Cloud QNN**:
   - Objective: Maximize sum rate through optimal assignment
   - Loss: Squared difference from target performance
   - Update: Gradient-free optimization (finite differences)

2. **Edge QNN**:
   - Objective: Maximize local sum rate given assignment
   - Loss: Reward-based loss (Eq. 17, 18)
   - Update: Local gradient descent

### Computational Complexity

- **Cloud QNN**: O(M × N × K) for M APs, N users, K antennas
- **Edge QNN**: O(K × N_local) per AP
- **Total Training**: O(I_cloud × I_edge) iterations

## 🎓 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{qnn_wireless_2024,
  title={Multi-Stage Quantum Neural Networks for Wireless Network Optimization},
  author={[Authors from paper]},
  journal={IEEE Transactions on Wireless Communications},
  year={2024}
}
```

## 📝 Notes

### Assumptions and Simplifications

1. **Perfect CSI**: Channel state information is assumed perfect
2. **Synchronized System**: All APs and users are synchronized
3. **Static Channels**: Channels are static during training/deployment
4. **Simplified Interference**: Interference model is simplified for tractability
5. **Ideal Quantum Gates**: Quantum gates are assumed noiseless

### Limitations

- Simulation uses quantum simulator (not real quantum hardware)
- Scalability limited by number of qubits
- Training time increases with network size
- Quantum-classical interface introduces overhead

## 🔮 Future Enhancements

Potential improvements:

- [ ] Implement on real quantum hardware (IBM Quantum, etc.)
- [ ] Add noise models for realistic quantum circuits
- [ ] Implement advanced optimizers (QAOA, VQE)
- [ ] Support for massive MIMO systems
- [ ] Dynamic channel adaptation
- [ ] Multi-objective optimization
- [ ] Integration with network simulators (ns-3, etc.)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is provided for educational and research purposes.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Last Updated**: February 2026

**Version**: 1.0.0

**Status**: Active Development
