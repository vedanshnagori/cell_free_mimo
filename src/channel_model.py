"""
Channel Model for Wireless Network
Implements channel generation and path loss calculations
"""

import numpy as np
from typing import Tuple, List
from config import NetworkConfig

class WirelessChannel:
    """Generates and manages wireless channel information"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        np.random.seed(config.RANDOM_SEED)
        
        # Generate random positions for APs and users
        self.ap_positions = self._generate_ap_positions()
        self.user_positions = self._generate_user_positions()
        
    def _generate_ap_positions(self) -> np.ndarray:
        """Generate positions for Access Points in a grid"""
        # Place APs in a grid pattern
        aps_per_side = int(np.ceil(np.sqrt(self.config.NUM_ACCESS_POINTS)))
        x_positions = np.linspace(20, self.config.AREA_WIDTH - 20, aps_per_side)
        y_positions = np.linspace(20, self.config.AREA_HEIGHT - 20, aps_per_side)
        
        positions = []
        for i in range(self.config.NUM_ACCESS_POINTS):
            x_idx = i % aps_per_side
            y_idx = i // aps_per_side
            positions.append([x_positions[x_idx], y_positions[y_idx]])
        
        return np.array(positions[:self.config.NUM_ACCESS_POINTS])
    
    def _generate_user_positions(self) -> np.ndarray:
        """Generate random positions for users"""
        x_positions = np.random.uniform(0, self.config.AREA_WIDTH, 
                                       self.config.NUM_USERS)
        y_positions = np.random.uniform(0, self.config.AREA_HEIGHT, 
                                       self.config.NUM_USERS)
        return np.column_stack([x_positions, y_positions])
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distances between all APs and users"""
        distances = np.zeros((self.config.NUM_ACCESS_POINTS, self.config.NUM_USERS))
        
        for i, ap_pos in enumerate(self.ap_positions):
            for j, user_pos in enumerate(self.user_positions):
                distances[i, j] = np.linalg.norm(ap_pos - user_pos)
        
        return distances
    
    def calculate_path_loss(self, distances: np.ndarray) -> np.ndarray:
        """Calculate path loss based on distance"""
        # Path loss in dB: PL = PL0 + 10*n*log10(d/d0)
        d0 = 1.0  # reference distance (1 meter)
        PL0 = 40  # path loss at reference distance (dB)
        
        path_loss_db = PL0 + 10 * self.config.PATH_LOSS_EXPONENT * np.log10(distances / d0)
        
        # Convert to linear scale
        path_loss_linear = 10 ** (path_loss_db / 10)
        return path_loss_linear
    
    def generate_channel_matrix(self) -> np.ndarray:
        """
        Generate channel matrix H with dimensions 
        [NUM_ACCESS_POINTS, NUM_USERS, NUM_ANTENNAS]
        
        Returns Rayleigh fading channel coefficients
        """
        # Calculate distances and path loss
        distances = self.calculate_distance_matrix()
        path_loss = self.calculate_path_loss(distances)
        
        # Generate Rayleigh fading (complex Gaussian)
        # Shape: [APs, Users, Antennas]
        rayleigh_real = np.random.randn(
            self.config.NUM_ACCESS_POINTS, 
            self.config.NUM_USERS, 
            self.config.NUM_ANTENNAS
        )
        rayleigh_imag = np.random.randn(
            self.config.NUM_ACCESS_POINTS, 
            self.config.NUM_USERS, 
            self.config.NUM_ANTENNAS
        )
        
        rayleigh = (rayleigh_real + 1j * rayleigh_imag) / np.sqrt(2)
        
        # Apply path loss
        channel_matrix = rayleigh / np.sqrt(path_loss[:, :, np.newaxis])
        
        return channel_matrix
    
    def get_channel_features(self, channel_matrix: np.ndarray) -> np.ndarray:
        """
        Extract features from channel matrix for QNN input
        Returns magnitude and phase information
        """
        # Flatten channel matrix and extract magnitude
        magnitude = np.abs(channel_matrix).flatten()
        phase = np.angle(channel_matrix).flatten()
        
        # Normalize features to [0, 1] or [-1, 1] for QNN input
        magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)
        phase_norm = phase / np.pi  # Phase normalized to [-1, 1]
        
        # Combine features
        features = np.concatenate([magnitude_norm, phase_norm])
        
        return features
    
    def generate_multiple_realizations(self, num_realizations: int) -> List[np.ndarray]:
        """Generate multiple channel realizations"""
        channels = []
        for _ in range(num_realizations):
            channels.append(self.generate_channel_matrix())
        return channels
    
    def get_network_info(self) -> dict:
        """Return network topology information"""
        return {
            'ap_positions': self.ap_positions,
            'user_positions': self.user_positions,
            'num_aps': self.config.NUM_ACCESS_POINTS,
            'num_users': self.config.NUM_USERS,
            'num_antennas': self.config.NUM_ANTENNAS
        }
