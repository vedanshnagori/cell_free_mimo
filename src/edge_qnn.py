"""
Edge QNN Module - Algorithm 3
Implements the Transmit Precoding Optimization using edge QNNs
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from typing import Dict, List
from config import NetworkConfig, QNNConfig

class EdgeQNN:
    """
    Edge QNN for transmit precoding optimization
    Implements Algorithm 3 from the paper
    Each AP has its own edge QNN
    """
    
    def __init__(self, ap_id: int, num_antennas: int, config: QNNConfig):
        self.ap_id = ap_id
        self.num_antennas = num_antennas
        self.config = config
        self.num_qubits = config.NUM_QUBITS_EDGE
        
        # Initialize parameters
        self.theta_edge = None
        self.trained = False
        
        # Setup quantum circuit
        self._setup_circuit()
        
    def _setup_circuit(self):
        """Setup the quantum circuit for edge QNN"""
        # Feature map for encoding local channel information
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=2,
            entanglement='linear'
        )
        
        # Variational form
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.config.REPS,
            entanglement=self.config.ENTANGLEMENT
        )
        
    def encode_local_channel(self, local_channel: np.ndarray, 
                            assignment: np.ndarray) -> np.ndarray:
        """
        Encode local channel information h_m and assignment γ
        Corresponds to the input of U^[m] in Eq. (19)
        
        Args:
            local_channel: Channel from this AP to its assigned users
            assignment: Assignment policy for this AP
        """
        # Extract relevant channel info and convert complex to real
        if np.iscomplexobj(local_channel):
            channel_magnitude = np.abs(local_channel).flatten()
            channel_phase = np.angle(local_channel).flatten()
            channel_features = np.concatenate([channel_magnitude, channel_phase])
        else:
            channel_features = np.abs(local_channel).flatten()
        
        assignment_features = assignment.flatten()
        
        # Combine features
        combined = np.concatenate([channel_features, assignment_features])
        
        # Truncate or pad to num_qubits
        if len(combined) < self.num_qubits:
            encoded = np.pad(combined, (0, self.num_qubits - len(combined)))
        else:
            encoded = combined[:self.num_qubits]
        
        # Normalize to [-π, π] and ensure real values
        encoded = np.real(encoded) * np.pi
        
        return encoded
    
    def create_qnn_circuit(self, input_data: np.ndarray, 
                          parameters: np.ndarray) -> QuantumCircuit:
        """
        Create the edge QNN circuit U^[m]
        Implements Eq. (19) from the paper
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Encoding layer (Eq. 20 - encoding operation)
        param_dict = dict(zip(self.feature_map.parameters, input_data))
        feature_circuit = self.feature_map.assign_parameters(param_dict)
        feature_circuit = feature_circuit.decompose()
        qc.compose(feature_circuit, inplace=True)
        
        # Variational layer (Eq. 20 - connection operation)
        param_dict = dict(zip(self.ansatz.parameters, parameters))
        ansatz_circuit = self.ansatz.assign_parameters(param_dict)
        ansatz_circuit = ansatz_circuit.decompose()
        qc.compose(ansatz_circuit, inplace=True)
        
        # Measurements
        qc.measure_all()
        
        return qc
    
    def decode_precoding(self, counts: Dict[str, int], 
                        num_users_assigned: int) -> np.ndarray:
        """
        Decode quantum output to precoding vector v_m
        
        Args:
            counts: Measurement results
            num_users_assigned: Number of users assigned to this AP
        
        Returns:
            Precoding vector (complex values)
        """
        # Convert counts to probabilities
        total_shots = sum(counts.values())
        
        # Initialize precoding matrix [num_antennas x max(1, num_users_assigned)]
        # Ensure at least 1 column even if no users assigned
        num_cols = max(1, num_users_assigned)
        precoding = np.zeros((self.num_antennas, num_cols), dtype=complex)
        
        # Decode measurement outcomes to precoding coefficients
        for bitstring, count in counts.items():
            prob = count / total_shots
            state_int = int(bitstring[::-1], 2)
            
            # Map quantum state to precoding coefficients
            for ant_idx in range(self.num_antennas):
                user_idx = ant_idx % num_cols
                
                # Extract phase and amplitude from quantum state
                phase = (state_int & 0xFF) * 2 * np.pi / 256
                amplitude = np.sqrt(prob)
                
                precoding[ant_idx, user_idx] += amplitude * np.exp(1j * phase)
        
        # Normalize precoding vectors
        for user_idx in range(num_cols):
            norm = np.linalg.norm(precoding[:, user_idx])
            if norm > 1e-10:
                precoding[:, user_idx] /= norm
            else:
                # If norm is too small, use random unit vector
                precoding[:, user_idx] = np.random.randn(self.num_antennas) + 1j * np.random.randn(self.num_antennas)
                precoding[:, user_idx] /= np.linalg.norm(precoding[:, user_idx])
        
        return precoding
    
   def calculate_precoding_quality(self, 
                                precoding: np.ndarray,
                                local_channel: np.ndarray,
                                assignment: np.ndarray,
                                all_precodings: list,      # added
                                all_channels: list,        # added
                                interference_factor: float,# added μ_n,k
                                snr: float = 10.0          # added ρ
                                ) -> float:
    quality = 0.0
    assigned_user_indices = np.where(assignment > 0.5)[0]
    num_users_assigned = len(assigned_user_indices)

    if num_users_assigned == 0:
        return 0.0

    num_aps = len(all_precodings)

    for idx, user_idx in enumerate(assigned_user_indices):

        if user_idx >= local_channel.shape[1]:
            continue

        # Channel and precoding for this AP-user pair
        h = local_channel[:, user_idx]
        v = precoding[:, idx] if idx < precoding.shape[1] else precoding[:, 0]

        min_len = min(len(h), len(v))
        h, v = h[:min_len], v[:min_len]

        # ── Signal power: ρ|h^T_m,k · v_m|²  (numerator of Eq. 4) ──
        signal_power = snr * (np.abs(np.dot(h.conj(), v)) ** 2)

        # ── Interference: ρ Σ_{n≠m} μ_n,k |h^T_n,k · v_n|²  (Eq. 4) ──
        interference = 0.0
        for n in range(num_aps):
            if n == self.ap_index:        # skip self
                continue

            if (n < len(all_channels) and 
                user_idx < all_channels[n].shape[1] and
                all_precodings[n] is not None):

                h_n = all_channels[n][:, user_idx]
                v_n = all_precodings[n]

                # Handle shape: v_n may be matrix, take first column
                if v_n.ndim > 1:
                    v_n = v_n[:, 0]

                min_len_n = min(len(h_n), len(v_n))
                h_n = h_n[:min_len_n]
                v_n = v_n[:min_len_n]

                interference += interference_factor * snr * \
                                (np.abs(np.dot(h_n.conj(), v_n)) ** 2)

        # ── SINR and Rate (Eq. 4 & 5) ──
        noise_power = 1.0                 # normalized σ² = 1
        sinr        = signal_power / (interference + noise_power)
        rate        = np.log2(1 + sinr)
        quality    += rate

    return quality
    
    def calculate_loss(self, precoding: np.ndarray,
                      local_channel: np.ndarray,
                      assignment: np.ndarray,
                      target_reward: float) -> float:
        """
        Calculate training loss L_precode
        Corresponds to Eq. (17) in the paper
        """
        current_quality = self.calculate_precoding_quality(
            precoding, local_channel, assignment
        )
        
        # Reward-based loss (Eq. 18)
        # φ_precode(h_m) = sum log2(1 + λ_j^[m] / σ_j^[m])
        loss = (target_reward - current_quality) ** 2
        
        return loss
    
    def train(self, local_channel: np.ndarray, assignment: np.ndarray,
             num_iterations: int = 30) -> Dict:
        """
        Train the Edge QNN for this AP (Algorithm 3)
        
        Args:
            local_channel: Local channel information for this AP
            assignment: Assignment policy γ from cloud QNN
            num_iterations: Number of training iterations
        """
        print(f"Training Edge QNN for AP {self.ap_id}...")
        
        # Initialize parameters
        num_params = self.ansatz.num_parameters
        self.theta_edge = np.random.uniform(0, 2*np.pi, num_params)
        
        # Determine number of assigned users for this AP
        if assignment.ndim == 2:
            # assignment is [num_aps x num_users]
            ap_assignment = assignment[self.ap_id, :]
        else:
            # assignment is [num_users]
            ap_assignment = assignment
        
        num_users_assigned = int(ap_assignment.sum())
        
        if num_users_assigned == 0:
            print(f"  No users assigned to AP {self.ap_id}, skipping training")
            self.trained = True
            return {'losses': [], 'qualities': []}
        
        # Target reward
        target_reward = 5.0 * num_users_assigned
        
        # Setup simulator
        simulator = AerSimulator()
        
        # Training history
        history = {
            'losses': [],
            'qualities': [],
            'precodings': []
        }
        
        # Training loop (Algorithm 3, step 2)
        for iteration in range(num_iterations):
            # Encode local channel and assignment (step 3)
            if assignment.ndim == 2:
                ap_assignment = assignment[self.ap_id, :]
            else:
                ap_assignment = assignment
            encoded_input = self.encode_local_channel(local_channel, ap_assignment)
            
            # Create and run circuit (step 4)
            qc = self.create_qnn_circuit(encoded_input, self.theta_edge)
            result = simulator.run(qc, shots=self.config.SHOTS).result()
            counts = result.get_counts()
            
            # Decode to get precoding v_m (step 4)
            precoding = self.decode_precoding(counts, num_users_assigned)
            
            # Calculate quality and loss (step 5)
            quality = self.calculate_precoding_quality(precoding, local_channel, assignment)
            loss = self.calculate_loss(precoding, local_channel, assignment, target_reward)
            
            # Store history
            history['losses'].append(loss)
            history['qualities'].append(quality)
            history['precodings'].append(precoding)
            
            # Parameter update (step 6)
            gradient = self._estimate_gradient(
                encoded_input, local_channel, assignment, target_reward, num_users_assigned
            )
            self.theta_edge = self.theta_edge - self.config.LEARNING_RATE * gradient
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Loss = {loss:.4f}, Quality = {quality:.4f}")
        
        self.trained = True
        print(f"Edge QNN training completed for AP {self.ap_id}!")
        
        return history
    
    def _estimate_gradient(self, encoded_input: np.ndarray,
                          local_channel: np.ndarray,
                          assignment: np.ndarray,
                          target_reward: float,
                          num_users_assigned: int) -> np.ndarray:
        """Estimate gradient for parameter update"""
        gradient = np.zeros_like(self.theta_edge)
        epsilon = 0.1
        
        simulator = AerSimulator()
        
        for i in range(len(self.theta_edge)):
            # Forward perturbation
            theta_plus = self.theta_edge.copy()
            theta_plus[i] += epsilon
            
            qc_plus = self.create_qnn_circuit(encoded_input, theta_plus)
            qc_plus = transpile(qc_plus, simulator)
            result_plus = simulator.run(qc_plus, shots=self.config.SHOTS).result()
            counts_plus = result_plus.get_counts()
            precoding_plus = self.decode_precoding(counts_plus, num_users_assigned)
            loss_plus = self.calculate_loss(precoding_plus, local_channel, 
                                          assignment, target_reward)
            
            # Backward perturbation
            theta_minus = self.theta_edge.copy()
            theta_minus[i] -= epsilon
            
            qc_minus = self.create_qnn_circuit(encoded_input, theta_minus)
            qc_minus = transpile(qc_minus, simulator)
            result_minus = simulator.run(qc_minus, shots=self.config.SHOTS).result()
            counts_minus = result_minus.get_counts()
            precoding_minus = self.decode_precoding(counts_minus, num_users_assigned)
            loss_minus = self.calculate_loss(precoding_minus, local_channel,
                                           assignment, target_reward)
            
            # Finite difference
            gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradient
    
    def predict(self, local_channel: np.ndarray, assignment: np.ndarray) -> np.ndarray:
        """
        Use trained edge QNN to predict precoding vectors
        Deployment phase
        """
        if not self.trained:
            raise ValueError(f"Edge QNN for AP {self.ap_id} must be trained before prediction!")
        
        num_users_assigned = int(assignment[self.ap_id].sum())
        
        if num_users_assigned == 0:
            return np.zeros((self.num_antennas, 1), dtype=complex)
        
        # Encode input
        encoded_input = self.encode_local_channel(local_channel, assignment[self.ap_id])
        
        # Run circuit
        simulator = AerSimulator()
        qc = self.create_qnn_circuit(encoded_input, self.theta_edge)
        qc = transpile(qc, simulator)
        result = simulator.run(qc, shots=self.config.SHOTS).result()
        counts = result.get_counts()
        
        # Decode precoding
        precoding = self.decode_precoding(counts, num_users_assigned)
        
        return precoding
