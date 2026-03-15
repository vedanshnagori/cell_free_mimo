"""
Channel Model for Wireless Network
Implements Eq. (3) from paper:
h_m,k = 1/√N_path * Σ g^[j]_m,k * a(φ_n)
g^[j]_m,k ~ CN(0, d^(-κ)_m,k)  (Section V)
"""

import numpy as np
from typing import Tuple, List, Optional
from config import NetworkConfig

class WirelessChannel:
    """
    Generates and manages wireless channel information
    Implements ULA channel model from Eq. (3)
    """

    def __init__(self, config: NetworkConfig):
        self.config = config

        # local seeded rng — avoids global seed side effects
        self.rng = np.random.default_rng(config.RANDOM_SEED)

        # Generate positions
        self.ap_positions   = self._generate_ap_positions()
        self.user_positions = self._generate_user_positions()

    # ── Position Generation ────────────────────────────────────────

    def _generate_ap_positions(self) -> np.ndarray:
        """Place APs in grid pattern across deployment area"""
        aps_per_side = int(np.ceil(np.sqrt(self.config.NUM_ACCESS_POINTS)))
        x_positions  = np.linspace(20, self.config.AREA_WIDTH  - 20, aps_per_side)
        y_positions  = np.linspace(20, self.config.AREA_HEIGHT - 20, aps_per_side)

        positions = []
        for i in range(self.config.NUM_ACCESS_POINTS):
            x_idx = i % aps_per_side
            y_idx = i // aps_per_side
            positions.append([x_positions[x_idx], y_positions[y_idx]])

        return np.array(positions[:self.config.NUM_ACCESS_POINTS])

    def _generate_user_positions(self) -> np.ndarray:
        """Generate random user positions within deployment area"""
        x = self.rng.uniform(0, self.config.AREA_WIDTH,  self.config.NUM_USERS)
        y = self.rng.uniform(0, self.config.AREA_HEIGHT, self.config.NUM_USERS)
        return np.column_stack([x, y])

    # ── Distance and Path Loss ─────────────────────────────────────

    def calculate_distance_matrix(self) -> np.ndarray:
        """Calculate normalized distances d_m,k ∈ [0,1] (Section V)"""
        distances = np.zeros((self.config.NUM_ACCESS_POINTS,
                              self.config.NUM_USERS))
        for i, ap_pos in enumerate(self.ap_positions):
            for j, user_pos in enumerate(self.user_positions):
                distances[i, j] = np.linalg.norm(ap_pos - user_pos)

        # normalize to [0,1] as per paper Section V
        max_dist = np.max(distances)
        return np.clip(distances / max_dist, 1e-10, 1.0)

    def calculate_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """
        Path loss from paper Section V:
        g^[j]_m,k ~ CN(0, d^(-κ)_m,k)
        path loss = d^(-κ)_m,k
        κ = PATH_LOSS_EXPONENT = 2.3 (Table III)
        """
        return distances ** (-self.config.PATH_LOSS_EXPONENT)

    # ── ULA Steering Vector ────────────────────────────────────────

    def _generate_ula_steering(self, angle: float) -> np.ndarray:
        """
        ULA steering vector a(φ_n) from Eq. (3):
        a(φ_n) = {exp(-i2πφ_n z)}_{z∈Z}
        Z = {n - 0.5*(N_Tx-1)}_{n∈{1,...,N_Tx}}
        """
        N_Tx = self.config.NUM_ANTENNAS
        Z    = np.arange(N_Tx) - 0.5 * (N_Tx - 1)
        return np.exp(-1j * 2 * np.pi * angle * Z)

    # ── Channel Generation ─────────────────────────────────────────

    def generate_channel_matrix(self,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> np.ndarray:
        """
        Generate channel matrix per Eq. (3):
        h_m,k = 1/√N_path * Σ_{n=1}^{N_path} g^[j]_m,k * a(φ_n)
        g^[j]_m,k ~ CN(0, d^(-κ)_m,k)

        Returns:
            channel: shape (N_AP, N_user, N_Tx) complex
        """
        rng       = rng or self.rng
        distances = self.calculate_distance_matrix()
        path_loss = self.calculate_path_loss(distances)
        N_path    = self.config.NUM_PATHS

        channel = np.zeros((self.config.NUM_ACCESS_POINTS,
                            self.config.NUM_USERS,
                            self.config.NUM_ANTENNAS), dtype=complex)

        for m in range(self.config.NUM_ACCESS_POINTS):
            for k in range(self.config.NUM_USERS):

                h_mk = np.zeros(self.config.NUM_ANTENNAS, dtype=complex)

                for n in range(N_path):
                    # path gain: g ~ CN(0, d^(-κ))
                    variance = path_loss[m, k]
                    g = np.sqrt(variance / 2) * (
                        rng.standard_normal() +
                        1j * rng.standard_normal()
                    )
                    # random departure angle φ_n ∈ [-0.5, 0.5]
                    phi_n = rng.uniform(-0.5, 0.5)

                    # ULA steering vector a(φ_n) — Eq. (3)
                    a = self._generate_ula_steering(phi_n)

                    h_mk += g * a

                # normalize by √N_path (Eq. 3)
                channel[m, k, :] = h_mk / np.sqrt(N_path)

        return channel

    def add_csi_imperfection(self, channel: np.ndarray,
                              noise_var: float = 0.01) -> np.ndarray:
        """
        Add CSI imperfection noise (Section III):
        Ĥ = H + n_CSI
        """
        n_csi = np.sqrt(noise_var / 2) * (
            self.rng.standard_normal(channel.shape) +
            1j * self.rng.standard_normal(channel.shape)
        )
        return channel + n_csi

    # ── Feature Extraction ─────────────────────────────────────────

    def get_channel_features(self, channel_matrix: np.ndarray) -> np.ndarray:
        """
        Extract normalized features for QNN input
        Consistent with encode_local_channel() normalization
        """
        magnitude      = np.abs(channel_matrix).flatten()
        magnitude_norm = magnitude / (np.max(magnitude) + 1e-10)  # → [0,1]

        phase      = np.angle(channel_matrix).flatten()
        phase_norm = phase / np.pi                                  # → [-1,1]

        return np.concatenate([magnitude_norm, phase_norm])

    def generate_multiple_realizations(self,
                                        num_realizations: int
                                        ) -> List[np.ndarray]:
        """
        Generate multiple reproducible channel realizations
        Each realization gets Ĥ = H + n_CSI (Section III)
        """
        # fresh rng for reproducibility regardless of call order
        rng      = np.random.default_rng(self.config.RANDOM_SEED)
        channels = []

        for _ in range(num_realizations):
            H     = self.generate_channel_matrix(rng)
            H_hat = self.add_csi_imperfection(H)   # Ĥ = H + n_CSI
            channels.append(H_hat)

        return channels

    def get_network_info(self) -> dict:
        """Return network topology information"""
        return {
            'ap_positions'  : self.ap_positions,
            'user_positions': self.user_positions,
            'num_aps'       : self.config.NUM_ACCESS_POINTS,
            'num_users'     : self.config.NUM_USERS,
            'num_antennas'  : self.config.NUM_ANTENNAS,
            'num_paths'     : self.config.NUM_PATHS
        }
