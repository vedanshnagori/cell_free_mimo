"""
Edge QNN Module - Algorithm 3
Implements the Transmit Precoding Optimization using edge QNNs
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap, RealAmplitudes
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
        self.ap_id        = ap_id
        self.ap_index     = ap_id
        self.num_antennas = num_antennas
        self.config       = config
        self.num_qubits   = config.NUM_QUBITS_EDGE

        # Setup quantum circuit first (needed for parameter count)
        self._setup_circuit()

        # Initialize parameters AFTER _setup_circuit()
        rng = np.random.default_rng(config.RANDOM_SEED)
        self.theta_edge = rng.uniform(
            low  = -np.pi,
            high =  np.pi,
            size =  self.ansatz.num_parameters
        )

        # Training state
        self.trained          = False
        self.current_episode  = 0
        self.learning_rate    = config.LEARNING_RATE
        self.training_losses  = []

        # Neighboring AP info for interference calculation
        self.all_precodings = None
        self.all_channels   = None

        # Reused simulator instance
        self.simulator = AerSimulator()

    def _setup_circuit(self):
        """
        Setup the quantum circuit for edge QNN
        Implements U^[m] = U^[m]_connect · U^[m]_encode (Eq. 19)
        Edge QNN requires Nuser qubits (Lemma 1)
        """

        # Validate qubit count matches Lemma 1
        assert self.num_qubits == self.config.NUM_QUBITS_EDGE, \
            f"Edge qubit count {self.num_qubits} != " \
            f"config {self.config.NUM_QUBITS_EDGE} (Lemma 1: needs Nuser qubits)"

        # Feature map selection from config
        feature_map_options = {
            'ZZFeatureMap'   : ZZFeatureMap,
            'ZFeatureMap'    : ZFeatureMap,
            'PauliFeatureMap': PauliFeatureMap
        }
        feature_map_class = feature_map_options.get(
            self.config.FEATURE_MAP,
            ZZFeatureMap
        )
        self.feature_map = feature_map_class(
            feature_dimension = self.num_qubits,
            reps              = self.config.REPS,
            entanglement      = self.config.ENTANGLEMENT
        )

        # Variational ansatz U^[m]_connect(θ^[m]) — Eq. (12)
        self.ansatz = RealAmplitudes(
            num_qubits   = self.num_qubits,
            reps         = self.config.REPS,
            entanglement = self.config.ENTANGLEMENT
        )

        # Validate circuit setup
        assert self.feature_map.num_parameters > 0, \
            "Feature map has no parameters — check feature_dimension and reps"
        assert self.ansatz.num_parameters > 0, \
            "Ansatz has no trainable parameters — check num_qubits and reps"

    def encode_local_channel(self, local_channel: np.ndarray,
                             assignment: np.ndarray) -> np.ndarray:
        """
        Encode local channel information h_m and assignment γ
        Corresponds to input of U^[m] in Eq. (19)
        Input: {h_m,k, γ_m} → output: rotation angles ∈ [-π, π]

        Args:
            local_channel: Channel h_m ∈ C^(N_Tx × N_user)
            assignment:    Assignment γ_m ∈ {0,1}^N_user

        Returns:
            encoded: rotation angles ∈ [-π, π], shape (num_qubits,)
        """

        # Input validation
        assert local_channel.shape[0] == self.num_antennas, \
            f"Channel shape {local_channel.shape} incompatible " \
            f"with num_antennas {self.num_antennas}"
        assert len(assignment) > 0, \
            "Assignment vector is empty"

        # Step 1: Complex → Real features
        if np.iscomplexobj(local_channel):
            channel_magnitude = np.abs(local_channel).flatten()
            channel_magnitude = channel_magnitude / \
                               (np.max(channel_magnitude) + 1e-10)

            channel_phase = np.angle(local_channel).flatten()
            channel_phase = channel_phase / np.pi

            channel_features = np.concatenate([channel_magnitude, channel_phase])
        else:
            channel_features = np.abs(local_channel).flatten()
            channel_features = channel_features / \
                              (np.max(channel_features) + 1e-10)

        # Step 2: Assignment features
        assignment_features = np.clip(assignment.flatten(), 0.0, 1.0)

        # Step 3: Combine features
        combined = np.concatenate([channel_features, assignment_features])

        # Step 4: Dimensionality reduction to num_qubits
        if len(combined) < self.num_qubits:
            encoded = np.pad(combined, (0, self.num_qubits - len(combined)))
        else:
            chunk_size = max(1, len(combined) // self.num_qubits)
            encoded = np.array([
                np.mean(combined[i:i + chunk_size])
                for i in range(0, self.num_qubits * chunk_size, chunk_size)
            ])
            if len(encoded) < self.num_qubits:
                encoded = np.pad(encoded, (0, self.num_qubits - len(encoded)))
            encoded = encoded[:self.num_qubits]

        # Step 5: Scale to rotation angles [-π, π]
        max_val = np.max(np.abs(encoded))
        if max_val > 1e-10:
            encoded = encoded / max_val
        encoded = np.real(encoded) * np.pi

        return encoded

    def create_qnn_circuit(self, input_data: np.ndarray,
                           parameters: np.ndarray) -> QuantumCircuit:
        """
        Create the edge QNN circuit U^[m]
        Implements Eq. (19): U^[m] = U^[m]_connect(θ^[m]) · U^[m]_encode(Ĥ)

        Args:
            input_data:  encoded channel features
            parameters:  trainable weights θ^[m]

        Returns:
            qc: QuantumCircuit with measurements
        """

        # Guard: ensure circuit is set up
        assert self.feature_map is not None, \
            "feature_map is None — call _setup_circuit() first"
        assert self.ansatz is not None, \
            "ansatz is None — call _setup_circuit() first"

        # Input validation
        assert len(input_data) == len(self.feature_map.parameters), \
            f"input_data length {len(input_data)} != " \
            f"feature_map parameters {len(self.feature_map.parameters)}"
        assert len(parameters) == len(self.ansatz.parameters), \
            f"parameters length {len(parameters)} != " \
            f"ansatz parameters {len(self.ansatz.parameters)}"

        # Initialize circuit |0⟩^⊗N_user (Eq. 19)
        cr = ClassicalRegister(self.num_qubits, name='edge_output')
        qc = QuantumCircuit(self.num_qubits)
        qc.add_register(cr)

        # Encoding layer: U^[m]_encode(Ĥ)
        param_dict      = dict(zip(self.feature_map.parameters, input_data))
        feature_circuit = self.feature_map.assign_parameters(param_dict)
        if self.config.BACKEND != 'statevector_simulator':
            feature_circuit = feature_circuit.decompose()
        qc.compose(feature_circuit, inplace=True)
        qc.barrier()

        # Variational layer: U^[m]_connect(θ^[m])
        param_dict     = dict(zip(self.ansatz.parameters, parameters))
        ansatz_circuit = self.ansatz.assign_parameters(param_dict)
        if self.config.BACKEND != 'statevector_simulator':
            ansatz_circuit = ansatz_circuit.decompose()
        qc.compose(ansatz_circuit, inplace=True)
        qc.barrier()

        # Measurement
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc

    def decode_precoding(self, counts: Dict[str, int],
                         num_users_assigned: int) -> np.ndarray:
        """
        Decode quantum measurement output to precoding vector v_m
        Output satisfies energy constraint ||v_m||² ≤ 1 (Eq. 15b)

        Args:
            counts:             Qiskit measurement results
            num_users_assigned: number of users assigned to this AP

        Returns:
            precoding: v_m ∈ C^(N_Tx × N_assigned)
        """

        # Input validation
        total_shots = sum(counts.values())
        assert len(counts)  > 0, \
            "counts is empty — circuit may have failed"
        assert total_shots  > 0, \
            "total_shots is zero — no measurements recorded"
        assert num_users_assigned >= 0, \
            f"num_users_assigned {num_users_assigned} cannot be negative"

        # Initialize precoding matrix
        num_cols  = max(1, num_users_assigned)
        precoding = np.zeros((self.num_antennas, num_cols), dtype=complex)
        min_prob  = 1.0 / self.config.SHOTS

        # Decode measurement outcomes
        for bitstring, count in counts.items():
            prob = count / total_shots

            if prob < min_prob:
                continue

            state_int = int(bitstring[::-1], 2)

            for ant_idx in range(self.num_antennas):
                bit_value = (state_int >> (ant_idx % self.num_qubits)) & 1
                user_idx  = bit_value % num_cols
                amplitude = np.sqrt(prob * self.config.SNR)
                phase     = (state_int / (2 ** self.num_qubits)) * 2 * np.pi
                precoding[ant_idx, user_idx] += amplitude * np.exp(1j * phase)

        # Normalize: enforce ||v_m||² ≤ 1 (Eq. 15b)
        rng = np.random.default_rng(self.config.RANDOM_SEED)

        for user_idx in range(num_cols):
            norm = np.linalg.norm(precoding[:, user_idx])
            if norm > 1e-10:
                precoding[:, user_idx] /= norm
            else:
                fallback = (rng.standard_normal(self.num_antennas) +
                           1j * rng.standard_normal(self.num_antennas))
                precoding[:, user_idx] = fallback / np.linalg.norm(fallback)

        return precoding

    def calculate_precoding_quality(self,
                                    precoding: np.ndarray,
                                    local_channel: np.ndarray,
                                    assignment: np.ndarray,
                                    all_precodings: list,
                                    all_channels: list,
                                    interference_factor: float,
                                    snr: float = 10.0) -> float:
        """
        Compute Q^[m]_precode = -R_m→k (Eq. 16)
        Full SINR with inter-AP interference (Eq. 4)
        """

        quality               = 0.0
        assigned_user_indices = np.where(assignment > 0.5)[0]
        num_users_assigned    = len(assigned_user_indices)

        if num_users_assigned == 0:
            return 0.0

        num_aps = len(all_precodings)

        for idx, user_idx in enumerate(assigned_user_indices):

            if user_idx >= local_channel.shape[1]:
                continue

            # Channel and precoding for this AP-user pair
            h = local_channel[:, user_idx]
            v = precoding[:, idx] if idx < precoding.shape[1] \
                else precoding[:, 0]

            min_len = min(len(h), len(v))
            h, v    = h[:min_len], v[:min_len]

            # Signal power: ρ|h^T_m,k · v_m|² (Eq. 4)
            signal_power = snr * (np.abs(np.dot(h.conj(), v)) ** 2)

            # Interference: ρ Σ_{n≠m} μ_n,k |h^T_n,k · v_n|² (Eq. 4)
            interference = 0.0
            for n in range(num_aps):
                if n == self.ap_index:
                    continue
                if (n < len(all_channels) and
                        all_channels[n] is not None and
                        user_idx < all_channels[n].shape[1] and
                        all_precodings[n] is not None):

                    h_n = all_channels[n][:, user_idx]
                    v_n = all_precodings[n]
                    if v_n.ndim > 1:
                        v_n = v_n[:, 0]

                    min_len_n = min(len(h_n), len(v_n))
                    h_n = h_n[:min_len_n]
                    v_n = v_n[:min_len_n]

                    interference += interference_factor * snr * \
                                   (np.abs(np.dot(h_n.conj(), v_n)) ** 2)

            # SINR and Rate (Eq. 4 & 5)
            noise_power  = 1.0
            sinr         = signal_power / (interference + noise_power)
            rate         = np.log2(1 + sinr)
            quality     += rate

        return quality

    def calculate_loss(self, precoding: np.ndarray,
                       local_channel: np.ndarray,
                       assignment: np.ndarray,
                       all_precodings: list,
                       all_channels: list) -> float:
        """
        Calculate training loss L_precode
        Implements Eq. (17): L_precode = ||Q^[m]_precode - Φ_precode||²
        """

        # Input validation
        assert precoding.shape[0]     == self.num_antennas, \
            f"precoding shape {precoding.shape} incompatible " \
            f"with num_antennas {self.num_antennas}"
        assert local_channel.shape[0] == self.num_antennas, \
            f"channel shape {local_channel.shape} incompatible " \
            f"with num_antennas {self.num_antennas}"
        assert len(assignment) > 0, \
            "assignment vector is empty"

        # Current reward: Q^[m]_precode = -R_m→k (Eq. 16)
        current_quality = self.calculate_precoding_quality(
            precoding           = precoding,
            local_channel       = local_channel,
            assignment          = assignment,
            all_precodings      = all_precodings,
            all_channels        = all_channels,
            interference_factor = self.config.INTERFERENCE_FACTOR,
            snr                 = self.config.SNR
        )
        current_reward = -current_quality

        # Target reward: Φ_precode(h_m) from Eq. (18)
        gram_matrix   = local_channel @ local_channel.conj().T
        eigenvalues   = np.linalg.eigvalsh(gram_matrix)
        eigenvalues   = np.maximum(eigenvalues, 0)
        n_lambda      = max(1, len(eigenvalues))
        target_reward = -sum(
            np.log2(1 + lam / (n_lambda * self.config.SNR))
            for lam in eigenvalues
        )

        # Loss: Eq. (17)
        loss = (current_reward - target_reward) ** 2
        self.training_losses.append(float(loss))

        return loss

    def train(self, local_channel: np.ndarray,
              assignment: np.ndarray,
              all_precodings: list,
              all_channels: list,
              num_iterations: int = None) -> Dict:
        """
        Train Edge QNN for this AP
        Implements Algorithm 3 from paper
        """

        num_iterations = num_iterations or self.config.NUM_ITERATIONS_EDGE
        print(f"Training Edge QNN for AP {self.ap_id}...")

        # Only initialize if not already set
        if self.theta_edge is None:
            rng = np.random.default_rng(self.config.RANDOM_SEED)
            self.theta_edge = rng.uniform(
                low  = -np.pi,
                high =  np.pi,
                size =  self.ansatz.num_parameters
            )

        # Extract this AP's assignment row
        ap_assignment = (assignment[self.ap_id, :]
                         if assignment.ndim == 2
                         else assignment)

        num_users_assigned = int(ap_assignment.sum())

        if num_users_assigned == 0:
            print(f"  No users assigned to AP {self.ap_id}, skipping")
            self.trained = True
            return {'losses': [], 'qualities': []}

        # Target reward: Φ_precode from Eq. (18)
        gram_matrix   = local_channel @ local_channel.conj().T
        eigenvalues   = np.maximum(np.linalg.eigvalsh(gram_matrix), 0)
        n_lambda      = max(1, len(eigenvalues))
        target_reward = -sum(
            np.log2(1 + lam / (n_lambda * self.config.SNR))
            for lam in eigenvalues
        )

        # Store neighboring info
        self.all_precodings = all_precodings
        self.all_channels   = all_channels

        history = {'losses': [], 'qualities': [], 'precodings': []}

        # Algorithm 3 training loop
        for iteration in range(num_iterations):

            # Step 3: encode {h_m, γ_m}
            encoded_input = self.encode_local_channel(
                local_channel, ap_assignment
            )

            # Step 4: run circuit → v_m
            qc        = self.create_qnn_circuit(encoded_input, self.theta_edge)
            counts    = self.simulator.run(
                qc, shots=self.config.SHOTS
            ).result().get_counts()
            precoding = self.decode_precoding(counts, num_users_assigned)

            # Step 5: compute Q^[m]_precode and L_precode
            quality = self.calculate_precoding_quality(
                precoding           = precoding,
                local_channel       = local_channel,
                assignment          = ap_assignment,
                all_precodings      = all_precodings,
                all_channels        = all_channels,
                interference_factor = self.config.INTERFERENCE_FACTOR,
                snr                 = self.config.SNR
            )
            loss = self.calculate_loss(
                precoding      = precoding,
                local_channel  = local_channel,
                assignment     = ap_assignment,
                all_precodings = all_precodings,
                all_channels   = all_channels
            )

            history['losses'].append(loss)
            history['qualities'].append(quality)
            history['precodings'].append(precoding)

            # Step 6: descending lr + parameter shift gradient
            lr       = self.config.LEARNING_RATE / \
                       np.sqrt(self.current_episode + 1)
            gradient = self._estimate_gradient(
                encoded_input, local_channel, ap_assignment,
                target_reward, num_users_assigned,
                all_precodings, all_channels
            )
            self.theta_edge = self.theta_edge - lr * gradient

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: "
                      f"Loss={loss:.4f}, "
                      f"Quality={quality:.4f}, "
                      f"LR={lr:.6f}")

        self.trained          = True
        self.current_episode += 1
        print(f"Edge QNN training completed for AP {self.ap_id}!")

        return history

    def _estimate_gradient(self, encoded_input: np.ndarray,
                           local_channel: np.ndarray,
                           assignment: np.ndarray,
                           target_reward: float,
                           num_users_assigned: int,
                           all_precodings: list,
                           all_channels: list) -> np.ndarray:
        """
        Parameter shift rule (Appendix B):
        ∇L(θ) = 1/(2sin(π/2)) * [L(θ+π/2) - L(θ-π/2)]
        Or Rotosolve if config.USE_ROTOSOLVE
        """

        gradient = np.zeros_like(self.theta_edge)
        shift    = np.pi / 2

        for i in range(len(self.theta_edge)):

            # Forward: θ + π/2
            theta_plus    = self.theta_edge.copy()
            theta_plus[i] += shift
            qc_plus       = self.create_qnn_circuit(encoded_input, theta_plus)
            if self.config.BACKEND != 'qasm_simulator':
                qc_plus = transpile(qc_plus, self.simulator)
            counts_plus    = self.simulator.run(
                qc_plus, shots=self.config.SHOTS
            ).result().get_counts()
            precoding_plus = self.decode_precoding(counts_plus, num_users_assigned)
            loss_plus      = self.calculate_loss(
                precoding      = precoding_plus,
                local_channel  = local_channel,
                assignment     = assignment,
                all_precodings = all_precodings,
                all_channels   = all_channels
            )

            # Backward: θ - π/2
            theta_minus    = self.theta_edge.copy()
            theta_minus[i] -= shift
            qc_minus       = self.create_qnn_circuit(encoded_input, theta_minus)
            if self.config.BACKEND != 'qasm_simulator':
                qc_minus = transpile(qc_minus, self.simulator)
            counts_minus    = self.simulator.run(
                qc_minus, shots=self.config.SHOTS
            ).result().get_counts()
            precoding_minus = self.decode_precoding(counts_minus, num_users_assigned)
            loss_minus      = self.calculate_loss(
                precoding      = precoding_minus,
                local_channel  = local_channel,
                assignment     = assignment,
                all_precodings = all_precodings,
                all_channels   = all_channels
            )

            if self.config.USE_ROTOSOLVE:
                # Rotosolve (Appendix B)
                theta_zero    = self.theta_edge.copy()
                theta_zero[i] = 0.0
                counts_zero   = self.simulator.run(
                    self.create_qnn_circuit(encoded_input, theta_zero),
                    shots=self.config.SHOTS
                ).result().get_counts()
                precoding_zero = self.decode_precoding(
                    counts_zero, num_users_assigned
                )
                loss_zero = self.calculate_loss(
                    precoding      = precoding_zero,
                    local_channel  = local_channel,
                    assignment     = assignment,
                    all_precodings = all_precodings,
                    all_channels   = all_channels
                )
                optimal_theta = -np.pi/2 - np.arctan2(
                    2 * loss_zero - loss_plus - loss_minus,
                    loss_plus - loss_minus
                )
                gradient[i] = self.theta_edge[i] - optimal_theta
            else:
                # Parameter shift rule (Appendix B)
                gradient[i] = (loss_plus - loss_minus) / \
                              (2 * np.sin(shift))

        # Gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        return gradient

    def predict(self, local_channel: np.ndarray,
                assignment: np.ndarray) -> np.ndarray:
        """
        Deployment phase inference for trained Edge QNN
        Algorithm 1 deployment step
        """

        if not self.trained:
            raise ValueError(
                f"Edge QNN for AP {self.ap_id} must be trained "
                f"before prediction"
            )

        assert self.theta_edge is not None, \
            f"theta_edge is None for AP {self.ap_id}"
        assert len(self.theta_edge) == self.ansatz.num_parameters, \
            f"theta_edge length {len(self.theta_edge)} != " \
            f"ansatz parameters {self.ansatz.num_parameters}"
        assert local_channel.shape[0] == self.num_antennas, \
            f"channel shape {local_channel.shape} incompatible " \
            f"with num_antennas {self.num_antennas}"
        assert assignment is not None, \
            "assignment cannot be None in deployment phase"

        # Extract this AP's assignment
        ap_assignment = (assignment[self.ap_id, :]
                         if assignment.ndim == 2
                         else assignment)

        num_users_assigned = int(ap_assignment.sum())

        if num_users_assigned == 0:
            print(f"  Warning: No users assigned to AP {self.ap_id} "
                  f"— returning zero precoding")
            return np.zeros((self.num_antennas, 1), dtype=complex)

        # Encode {h_m, γ_m}
        encoded_input = self.encode_local_channel(
            local_channel, ap_assignment
        )

        # Run trained circuit U^[m](θ^[m])
        qc = self.create_qnn_circuit(encoded_input, self.theta_edge)
        if self.config.BACKEND != 'qasm_simulator':
            qc = transpile(qc, self.simulator)

        counts = self.simulator.run(
            qc, shots=self.config.SHOTS
        ).result().get_counts()

        # Decode counts → v_m
        precoding = self.decode_precoding(counts, num_users_assigned)

        return precoding