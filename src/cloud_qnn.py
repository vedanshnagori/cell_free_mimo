"""
Cloud QNN Module - Algorithm 2
Implements the Transmitter-User Assignment using QNN
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from typing import Tuple, Dict
from config import NetworkConfig, QNNConfig

class CloudQNN:
    """
    Cloud QNN for transmitter-user assignment optimization
    Implements Algorithm 2 from the paper
    """
    
    def __init__(self, num_aps: int, num_users: int, config: QNNConfig):
        self.num_aps = num_aps
        self.num_users = num_users
        self.config = config
        self.num_qubits = config.NUM_QUBITS_CLOUD
        
        # Initialize parameters
        self.theta_cloud = None
        self.trained = False
        
        # Setup quantum circuit
        self._setup_circuit()
        
    def _setup_circuit(self):
        """Setup the quantum circuit architecture"""
        # Feature map for encoding channel information
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=2,
            entanglement='linear'
        )
        
        # Variational form (ansatz)
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.config.REPS,
            entanglement=self.config.ENTANGLEMENT
        )
        
        # Create parameter vectors
        self.input_params = ParameterVector('x', self.num_qubits)
        self.weight_params = ParameterVector('θ', self.ansatz.num_parameters)
        
    def encode_channel_info(self, channel_features: np.ndarray) -> np.ndarray:
        """
        Encode channel information H into quantum features
        Corresponds to Eq. (10) in the paper
        """
        # Convert complex to real by taking magnitude and phase separately
        if np.iscomplexobj(channel_features):
            magnitude = np.abs(channel_features)
            phase = np.angle(channel_features)
            # Combine magnitude and phase
            real_features = np.concatenate([magnitude.flatten(), phase.flatten()])
        else:
            real_features = channel_features.flatten()
        
        # Take first num_qubits features or pad if needed
        if len(real_features) < self.num_qubits:
            encoded = np.pad(real_features, (0, self.num_qubits - len(real_features)))
        else:
            encoded = real_features[:self.num_qubits]
        
        # Normalize to [-π, π] for angle encoding
        # Make sure all values are real
        encoded = np.real(encoded) * np.pi
        return encoded
    
    def create_qnn_circuit(self, input_data: np.ndarray, parameters: np.ndarray) -> QuantumCircuit:
        """
        Create the complete QNN circuit with encoding and variational layers
        Implements U^cloud in Eq. (10)
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Encoding layer - embed classical information into Hilbert space (Eq. 11)
        param_dict = dict(zip(self.feature_map.parameters, input_data))
        feature_circuit = self.feature_map.assign_parameters(param_dict)
        # Decompose to basic gates
        feature_circuit = feature_circuit.decompose()
        qc.compose(feature_circuit, inplace=True)
        
        # Variational layer - parameterized quantum gates
        param_dict = dict(zip(self.ansatz.parameters, parameters))
        ansatz_circuit = self.ansatz.assign_parameters(param_dict)
        # Decompose to basic gates
        ansatz_circuit = ansatz_circuit.decompose()
        qc.compose(ansatz_circuit, inplace=True)
        
        # Measurements
        qc.measure_all()
        
        return qc
    
    def decode_output(self, counts: Dict[str, int]) -> np.ndarray:
        """
        Decode quantum measurement results to assignment policy γ
        Implements decoding step in Algorithm 2, step 5
        """
        # Convert counts to probabilities
        total_shots = sum(counts.values())
        
        # Create assignment matrix (AP x User)
        assignment = np.zeros((self.num_aps, self.num_users))
        
        # Simple decoding: map measurement outcomes to assignments
        # Use first log2(num_aps * num_users) qubits
        for bitstring, count in counts.items():
            prob = count / total_shots
            # Decode bitstring to AP-User pair
            # This is a simplified decoding scheme
            state_int = int(bitstring[::-1], 2)  # Reverse for little-endian
            
            ap_idx = state_int % self.num_aps
            user_idx = (state_int // self.num_aps) % self.num_users
            
            if ap_idx < self.num_aps and user_idx < self.num_users:
                assignment[ap_idx, user_idx] += prob
        
        # Normalize to create valid assignment (each user assigned to exactly one AP)
        assignment = self._normalize_assignment(assignment)
        
        return assignment
    
    def _normalize_assignment(self, assignment: np.ndarray) -> np.ndarray:
        """Ensure each user is assigned to exactly one AP (constraint in paper)"""
        normalized = np.zeros_like(assignment)
        
        for user_idx in range(self.num_users):
            user_probs = assignment[:, user_idx]
            if user_probs.sum() > 0:
                # Assign user to AP with highest probability
                best_ap = np.argmax(user_probs)
                normalized[best_ap, user_idx] = 1.0
            else:
                # Random assignment if no clear winner
                random_ap = np.random.randint(0, self.num_aps)
                normalized[random_ap, user_idx] = 1.0
        
        return normalized
    
    def calculate_assignment_quality(self, assignment: np.ndarray, 
                                    channel_matrix: np.ndarray) -> float:
        """
        Calculate quality metric Q_assign for the assignment
        Corresponds to Eq. (14) in the paper
        """
        quality = 0.0
        
        # Sum rate calculation (simplified)
        for ap_idx in range(self.num_aps):
            for user_idx in range(self.num_users):
                if assignment[ap_idx, user_idx] > 0:
                    # Channel gain from AP to user
                    channel_gain = np.linalg.norm(channel_matrix[ap_idx, user_idx, :]) ** 2
                    # SINR approximation
                    sinr = channel_gain / (1 + 0.1)  # Simplified noise/interference
                    rate = np.log2(1 + sinr)
                    quality += assignment[ap_idx, user_idx] * rate
        
        return quality
    
    def calculate_loss(self, assignment: np.ndarray, channel_matrix: np.ndarray,
                      target_performance: float) -> float:
        """
        Calculate training loss L_assign
        Corresponds to Eq. (13) in the paper
        """
        current_quality = self.calculate_assignment_quality(assignment, channel_matrix)
        
        # Loss is difference from target (reference point)
        loss = (target_performance - current_quality) ** 2
        
        return loss
    
    def train(self, channel_data: np.ndarray, num_iterations: int = 50) -> Dict:
        """
        Train the Cloud QNN (Algorithm 2)
        
        Args:
            channel_data: Channel matrix realizations
            num_iterations: Number of training iterations
        """
        print("Training Cloud QNN for Transmitter-User Assignment...")
        
        # Initialize parameters randomly
        num_params = self.ansatz.num_parameters
        self.theta_cloud = np.random.uniform(0, 2*np.pi, num_params)
        
        # Target performance (reference point)
        target_performance = 10.0  # Example target sum rate
        
        # Setup simulator
        simulator = AerSimulator()
        sampler = Sampler()
        
        # Training history
        history = {
            'losses': [],
            'assignments': [],
            'qualities': []
        }
        
        # Training loop (Algorithm 2, step 3)
        for iteration in range(num_iterations):
            # Encode channel information (step 4)
            channel_features = channel_data.flatten()
            encoded_input = self.encode_channel_info(channel_features)
            
            # Create and run circuit
            qc = self.create_qnn_circuit(encoded_input, self.theta_cloud)
            qc_transpiled = transpile(qc, simulator)
            result = simulator.run(qc_transpiled, shots=self.config.SHOTS).result()
            counts = result.get_counts()
            
            # Decode to get assignment policy γ (step 5)
            assignment = self.decode_output(counts)
            
            # Calculate quality and loss (steps 6-7)
            quality = self.calculate_assignment_quality(assignment, channel_data)
            loss = self.calculate_loss(assignment, channel_data, target_performance)
            
            # Store history
            history['losses'].append(loss)
            history['assignments'].append(assignment)
            history['qualities'].append(quality)
            
            # Parameter update (gradient-free optimization)
            # In practice, use optimizer like COBYLA or SPSA
            gradient_estimate = self._estimate_gradient(
                encoded_input, channel_data, target_performance
            )
            self.theta_cloud = self.theta_cloud - self.config.LEARNING_RATE * gradient_estimate
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Loss = {loss:.4f}, Quality = {quality:.4f}")
        
        self.trained = True
        print("Cloud QNN training completed!")
        
        return history
    
    def _estimate_gradient(self, encoded_input: np.ndarray, 
                          channel_data: np.ndarray,
                          target_performance: float) -> np.ndarray:
        """Estimate gradient using parameter shift rule or finite differences"""
        gradient = np.zeros_like(self.theta_cloud)
        epsilon = 0.1
        
        simulator = AerSimulator()
        
        for i in range(len(self.theta_cloud)):
            # Forward perturbation
            theta_plus = self.theta_cloud.copy()
            theta_plus[i] += epsilon
            
            qc_plus = self.create_qnn_circuit(encoded_input, theta_plus)
            qc_plus = transpile(qc_plus, simulator)
            result_plus = simulator.run(qc_plus, shots=self.config.SHOTS).result()
            counts_plus = result_plus.get_counts()
            assignment_plus = self.decode_output(counts_plus)
            loss_plus = self.calculate_loss(assignment_plus, channel_data, target_performance)
            
            # Backward perturbation
            theta_minus = self.theta_cloud.copy()
            theta_minus[i] -= epsilon
            
            qc_minus = self.create_qnn_circuit(encoded_input, theta_minus)
            qc_minus = transpile(qc_minus, simulator)
            result_minus = simulator.run(qc_minus, shots=self.config.SHOTS).result()
            counts_minus = result_minus.get_counts()
            assignment_minus = self.decode_output(counts_minus)
            loss_minus = self.calculate_loss(assignment_minus, channel_data, target_performance)
            
            # Finite difference
            gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        return gradient
    
    def predict(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Use trained QNN to predict assignment policy
        Deployment phase (Algorithm 1, step 9)
        """
        if not self.trained:
            raise ValueError("Cloud QNN must be trained before prediction!")
        
        # Encode channel information
        channel_features = channel_data.flatten()
        encoded_input = self.encode_channel_info(channel_features)
        
        # Create and run circuit with trained parameters
        simulator = AerSimulator()
        qc = self.create_qnn_circuit(encoded_input, self.theta_cloud)
        qc = transpile(qc, simulator)
        result = simulator.run(qc, shots=self.config.SHOTS).result()
        counts = result.get_counts()
        
        # Decode to get assignment
        assignment = self.decode_output(counts)
        
        return assignment
