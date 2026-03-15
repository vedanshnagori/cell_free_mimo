"""
Cloud QNN Module - Algorithm 2
Implements Transmitter-User Assignment using QNN
U^cloud = U^cloud_connect(θ^cloud) · U^cloud_encode(Ĥ)  (Eq. 10)
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import (ZZFeatureMap, ZFeatureMap,
                                     PauliFeatureMap, RealAmplitudes)
from qiskit_aer import AerSimulator
from typing import Dict
from config import NetworkConfig, QNNConfig

class CloudQNN:
    """
    Cloud QNN for transmitter-user assignment
    Implements Algorithm 2 from paper
    """

    def __init__(self, num_aps: int, num_users: int, config: QNNConfig):
        self.num_aps    = num_aps
        self.num_users  = num_users
        self.config     = config
        self.num_qubits = config.NUM_QUBITS_CLOUD

        # seeded rng — consistent with NetworkConfig.RANDOM_SEED
        self.rng = np.random.default_rng(config.RANDOM_SEED)

        # Setup circuit first — needed for parameter count
        self._setup_circuit()

        # Initialize parameters AFTER _setup_circuit()
        self.theta_cloud = self.rng.uniform(
            low  = -np.pi,
            high =  np.pi,
            size =  self.ansatz.num_parameters
        )

        # Training state
        self.trained          = False
        self.current_episode  = 0
        self.learning_rate    = config.LEARNING_RATE
        self.training_losses  = []

        # Simulator instance — reused across calls
        self.simulator = AerSimulator()

    # ── Circuit Setup ──────────────────────────────────────────────

    def _setup_circuit(self):
        """
        Setup U^cloud = U^cloud_connect · U^cloud_encode
        Eq. (10), (11), (12)
        """
        # feature map selection from config
        feature_map_options = {
            'ZZFeatureMap'   : ZZFeatureMap,
            'ZFeatureMap'    : ZFeatureMap,
            'PauliFeatureMap': PauliFeatureMap
        }
        feature_map_class = feature_map_options.get(
            self.config.FEATURE_MAP, ZZFeatureMap
        )
        self.feature_map = feature_map_class(
            feature_dimension = self.num_qubits,
            reps              = self.config.REPS,
            entanglement      = self.config.ENTANGLEMENT
        )

        # variational ansatz U^cloud_connect(θ^cloud) — Eq. (12)
        self.ansatz = RealAmplitudes(
            num_qubits   = self.num_qubits,
            reps         = self.config.REPS,
            entanglement = self.config.ENTANGLEMENT
        )

        # validate
        assert self.feature_map.num_parameters > 0, \
            "Feature map has no parameters"
        assert self.ansatz.num_parameters > 0, \
            "Ansatz has no trainable parameters"

    # ── Encoding ───────────────────────────────────────────────────

    def encode_channel_info(self, channel_matrix: np.ndarray) -> np.ndarray:
        """
        Encode Ĥ = {Ĥ_m}^N_AP into quantum rotation angles
        Implements U^cloud_encode (Eq. 11)
        """
        if np.iscomplexobj(channel_matrix):
            magnitude = np.abs(channel_matrix).flatten()
            magnitude = magnitude / (np.max(magnitude) + 1e-10)  # → [0,1]

            phase = np.angle(channel_matrix).flatten()
            phase = phase / np.pi                                  # → [-1,1]

            real_features = np.concatenate([magnitude, phase])
        else:
            real_features = np.abs(channel_matrix).flatten()
            real_features = real_features / \
                           (np.max(real_features) + 1e-10)

        # mean pooling to num_qubits
        if len(real_features) < self.num_qubits:
            encoded = np.pad(
                real_features,
                (0, self.num_qubits - len(real_features))
            )
        else:
            chunk_size = max(1, len(real_features) // self.num_qubits)
            encoded = np.array([
                np.mean(real_features[i:i+chunk_size])
                for i in range(0, self.num_qubits*chunk_size, chunk_size)
            ])
            if len(encoded) < self.num_qubits:
                encoded = np.pad(
                    encoded,
                    (0, self.num_qubits - len(encoded))
                )
            encoded = encoded[:self.num_qubits]

        # normalize then scale to rotation angles [-π, π]
        max_val = np.max(np.abs(encoded))
        if max_val > 1e-10:
            encoded = encoded / max_val
        encoded = np.real(encoded) * np.pi

        return encoded

    # ── Circuit Creation ───────────────────────────────────────────

    def create_qnn_circuit(self, input_data: np.ndarray,
                           parameters: np.ndarray) -> QuantumCircuit:
        """
        Build U^cloud = U^cloud_connect(θ^cloud) · U^cloud_encode(Ĥ)
        Implements Eq. (10)
        """
        assert self.feature_map is not None, \
            "feature_map is None — call _setup_circuit() first"
        assert self.ansatz is not None, \
            "ansatz is None — call _setup_circuit() first"
        assert len(input_data) == len(self.feature_map.parameters), \
            f"input_data {len(input_data)} != " \
            f"feature_map params {len(self.feature_map.parameters)}"
        assert len(parameters) == len(self.ansatz.parameters), \
            f"parameters {len(parameters)} != " \
            f"ansatz params {len(self.ansatz.parameters)}"

        cr = ClassicalRegister(self.num_qubits, name='cloud_output')
        qc = QuantumCircuit(self.num_qubits)
        qc.add_register(cr)

        # U^cloud_encode — Eq. (11)
        param_dict      = dict(zip(self.feature_map.parameters, input_data))
        feature_circuit = self.feature_map.assign_parameters(param_dict)
        if self.config.BACKEND != 'statevector_simulator':
            feature_circuit = feature_circuit.decompose()
        qc.compose(feature_circuit, inplace=True)
        qc.barrier()

        # U^cloud_connect(θ^cloud) — Eq. (12)
        param_dict     = dict(zip(self.ansatz.parameters, parameters))
        ansatz_circuit = self.ansatz.assign_parameters(param_dict)
        if self.config.BACKEND != 'statevector_simulator':
            ansatz_circuit = ansatz_circuit.decompose()
        qc.compose(ansatz_circuit, inplace=True)
        qc.barrier()

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    # ── Decoding ───────────────────────────────────────────────────

    def decode_output(self, counts: Dict[str, int]) -> np.ndarray:
        """
        Decode measurement to assignment γ ∈ {0,1}^(N_AP × N_user)
        Implements Eq. (9): π_assign: Ĥ → γ
        o^cloud_m ∈ [0,1] → ô_m = floor(o^cloud_m * N_user)
        """
        total_shots = sum(counts.values())
        assignment  = np.zeros((self.num_aps, self.num_users))

        for m in range(self.num_aps):
            # probability of qubit m being |1⟩
            o_cloud_m = sum(
                count / total_shots
                for bitstring, count in counts.items()
                if bitstring[::-1][m % self.num_qubits] == '1'
            )
            # decode to user index: ô_m = floor(o^cloud_m * N_user)
            user_idx = int(np.floor(o_cloud_m * self.num_users))
            user_idx = min(user_idx, self.num_users - 1)
            assignment[m, user_idx] = 1.0

        return self._normalize_assignment(assignment)

    def _normalize_assignment(self,
                               assignment: np.ndarray) -> np.ndarray:
        """
        Enforce Eq. (6c): ϱ_k ≥ 1 — each user gets at least one AP
        Allows many-to-one: multiple APs can serve same user (Section IV-A)
        """
        normalized = (assignment > 0.5).astype(float)

        # enforce Eq. (6c) — add r_penalty for unserved users
        for user_idx in range(self.num_users):
            if normalized[:, user_idx].sum() == 0:
                best_ap = np.argmax(assignment[:, user_idx])
                normalized[best_ap, user_idx] = 1.0

        return normalized

    # ── Quality and Loss ───────────────────────────────────────────

    def calculate_assignment_quality(self,
                                      assignment: np.ndarray,
                                      channel_matrix: np.ndarray
                                      ) -> float:
        """
        Compute Q_assign = -min_k R_k (Eq. 14)
        Uses max-ratio precoding v^MR_m = ĥ*_m/||ĥ*_m|| (Section IV-A)
        """
        min_rate = float('inf')

        for k in range(self.num_users):
            assigned_aps = np.where(assignment[:, k] > 0.5)[0]

            if len(assigned_aps) == 0:
                # penalty for unassigned user — Eq. (14)
                return float(self.config.R_PENALTY)

            # signal: Σ_{m∈A_k} ρ|ĥ^T_mk v^MR_m|²
            signal = 0.0
            for m in assigned_aps:
                h_mk = channel_matrix[m, k, :]
                v_mr = h_mk.conj() / (np.linalg.norm(h_mk) + 1e-10)
                signal += self.config.SNR * \
                          np.abs(h_mk.conj() @ v_mr) ** 2

            # interference: ρ Σ_{n∉A_k} μ_nk |ĥ^T_nk v^MR_n|²
            interference = sum(
                self.config.INTERFERENCE_FACTOR * self.config.SNR *
                np.linalg.norm(channel_matrix[n, k, :]) ** 2
                for n in range(self.num_aps)
                if assignment[n, k] < 0.5
            )

            rate     = np.log2(1 + signal / (interference + 1.0))
            min_rate = min(min_rate, rate)

        # Q_assign = -min_k R_k  (Eq. 14)
        return -min_rate

    def calculate_loss(self, assignment: np.ndarray,
                       channel_matrix: np.ndarray) -> float:
        """
        L_assign = ||Q_assign - Φ_assign||²  (Eq. 13)
        Φ_assign = -Σ Σ log2(1 + λ^[m]_i/N_λ * ρ)  (Eq. 14)
        """
        # current reward Q_assign
        current_quality = self.calculate_assignment_quality(
            assignment, channel_matrix
        )

        # target Φ_assign from eigenvalues (Eq. 14)
        gram     = channel_matrix.reshape(-1, channel_matrix.shape[-1])
        gram     = gram @ gram.conj().T
        eigvals  = np.maximum(np.linalg.eigvalsh(gram), 0)
        n_lambda = max(1, len(eigvals))
        target   = -sum(
            np.log2(1 + lam / (n_lambda * self.config.SNR))
            for lam in eigvals
        )

        # L_assign = ||Q_assign - Φ_assign||²  (Eq. 13)
        loss = (current_quality - target) ** 2
        self.training_losses.append(float(loss))
        return loss

    # ── Training ───────────────────────────────────────────────────

    def train(self, channel_data: np.ndarray,
              num_iterations: int = None) -> Dict:
        """
        Train Cloud QNN — Algorithm 2
        Optimizes θ^cloud for transmitter-user assignment
        """
        num_iterations = num_iterations or self.config.NUM_ITERATIONS_CLOUD
        print("Training Cloud QNN for Transmitter-User Assignment...")

        # only reinitialize if not already set
        if self.theta_cloud is None:
            self.theta_cloud = self.rng.uniform(
                low  = -np.pi,
                high =  np.pi,
                size =  self.ansatz.num_parameters
            )

        history = {'losses': [], 'assignments': [], 'qualities': []}

        for iteration in range(num_iterations):

            # encode Ĥ = {Ĥ_m} → rotation angles (Eq. 11)
            encoded_input = self.encode_channel_info(channel_data)

            # run U^cloud circuit
            qc = self.create_qnn_circuit(encoded_input, self.theta_cloud)
            if self.config.BACKEND != 'qasm_simulator':
                qc = transpile(qc, self.simulator)

            counts     = self.simulator.run(
                qc, shots=self.config.SHOTS
            ).result().get_counts()

            # decode → γ (Eq. 9)
            assignment = self.decode_output(counts)

            # compute Q_assign and L_assign (Eq. 13, 14)
            quality = self.calculate_assignment_quality(
                assignment, channel_data
            )
            loss = self.calculate_loss(assignment, channel_data)

            history['losses'].append(loss)
            history['assignments'].append(assignment)
            history['qualities'].append(quality)

            # descending lr: μ/√(episode+1) (Section V)
            lr       = self.config.LEARNING_RATE / \
                       np.sqrt(self.current_episode + 1)
            gradient = self._estimate_gradient(
                encoded_input, channel_data
            )
            self.theta_cloud = self.theta_cloud - lr * gradient

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: "
                      f"Loss={loss:.4f}, "
                      f"Quality={quality:.4f}, "
                      f"LR={lr:.6f}")

        self.trained         = True
        self.current_episode += 1
        print("Cloud QNN training completed!")
        return history

    # ── Gradient ───────────────────────────────────────────────────

    def _estimate_gradient(self, encoded_input: np.ndarray,
                           channel_data: np.ndarray) -> np.ndarray:
        """
        Parameter shift rule (Appendix B):
        ∇L = 1/(2sin(π/2)) * [L(θ+π/2) - L(θ-π/2)]
        """
        gradient = np.zeros_like(self.theta_cloud)
        shift    = np.pi / 2

        for i in range(len(self.theta_cloud)):

            # L(θ + π/2)
            theta_plus    = self.theta_cloud.copy()
            theta_plus[i] += shift
            qc_plus  = self.create_qnn_circuit(encoded_input, theta_plus)
            counts_plus   = self.simulator.run(
                qc_plus, shots=self.config.SHOTS
            ).result().get_counts()
            loss_plus = self.calculate_loss(
                self.decode_output(counts_plus), channel_data
            )

            # L(θ - π/2)
            theta_minus    = self.theta_cloud.copy()
            theta_minus[i] -= shift
            qc_minus = self.create_qnn_circuit(encoded_input, theta_minus)
            counts_minus   = self.simulator.run(
                qc_minus, shots=self.config.SHOTS
            ).result().get_counts()
            loss_minus = self.calculate_loss(
                self.decode_output(counts_minus), channel_data
            )

            # parameter shift rule (Appendix B)
            gradient[i] = (loss_plus - loss_minus) / \
                          (2 * np.sin(shift))

            # Rotosolve (Appendix B) if enabled
            if self.config.USE_ROTOSOLVE:
                theta_zero    = self.theta_cloud.copy()
                theta_zero[i] = 0.0
                counts_zero   = self.simulator.run(
                    self.create_qnn_circuit(encoded_input, theta_zero),
                    shots=self.config.SHOTS
                ).result().get_counts()
                loss_zero = self.calculate_loss(
                    self.decode_output(counts_zero), channel_data
                )
                optimal   = -np.pi/2 - np.arctan2(
                    2*loss_zero - loss_plus - loss_minus,
                    loss_plus - loss_minus
                )
                gradient[i] = self.theta_cloud[i] - optimal

        # gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        return gradient

    # ── Prediction ─────────────────────────────────────────────────

    def predict(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Deployment phase — Algorithm 1 step 9
        Estimate γ using trained U^cloud(θ^cloud)
        """
        if not self.trained:
            raise ValueError(
                "Cloud QNN must be trained before prediction"
            )

        assert self.theta_cloud is not None, \
            "theta_cloud is None — training may have failed"
        assert len(self.theta_cloud) == self.ansatz.num_parameters, \
            f"theta_cloud length {len(self.theta_cloud)} != " \
            f"ansatz parameters {self.ansatz.num_parameters}"

        encoded_input = self.encode_channel_info(channel_data)
        qc = self.create_qnn_circuit(encoded_input, self.theta_cloud)

        if self.config.BACKEND != 'qasm_simulator':
            qc = transpile(qc, self.simulator)

        counts     = self.simulator.run(
            qc, shots=self.config.SHOTS
        ).result().get_counts()
        assignment = self.decode_output(counts)

        return assignment
