"""
Multi-Stage QNN System - Algorithm 1
Main implementation that orchestrates cloud and edge QNNs
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from config import NetworkConfig, QNNConfig
from channel_model import WirelessChannel
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN

class MultiStageQNN:
    """
    Multi-Stage QNN System for Wireless Network Optimization
    Implements Algorithm 1 from the paper
    """
    
    def __init__(self, network_config: NetworkConfig, qnn_config: QNNConfig):
        self.network_config = network_config
        self.qnn_config = qnn_config
        
        # Initialize channel model
        self.channel_model = WirelessChannel(network_config)
        
        # Initialize Cloud QNN for transmitter-user assignment (Algorithm 2)
        self.cloud_qnn = CloudQNN(
            num_aps=network_config.NUM_ACCESS_POINTS,
            num_users=network_config.NUM_USERS,
            config=qnn_config
        )
        
        # Initialize Edge QNNs for each AP (Algorithm 3)
        self.edge_qnns = []
        for ap_id in range(network_config.NUM_ACCESS_POINTS):
            edge_qnn = EdgeQNN(
                ap_id=ap_id,
                num_antennas=network_config.NUM_ANTENNAS,
                config=qnn_config
            )
            self.edge_qnns.append(edge_qnn)
        
        print(f"Multi-Stage QNN System initialized:")
        print(f"  - {network_config.NUM_ACCESS_POINTS} Access Points")
        print(f"  - {network_config.NUM_USERS} Users")
        print(f"  - {network_config.NUM_ANTENNAS} Antennas per AP")
        print(f"  - Cloud QNN: {qnn_config.NUM_QUBITS_CLOUD} qubits")
        print(f"  - Edge QNN: {qnn_config.NUM_QUBITS_EDGE} qubits per AP")
    
    def train_cloud_qnn(self) -> Dict:
        """
        Train Cloud QNN (Algorithm 1, step 2 + Algorithm 2)
        Cloud QNN training with complete channel information
        """
        print("\n" + "="*70)
        print("PHASE 1: CLOUD QNN TRAINING")
        print("="*70)
        
        # Generate complete channel information for the network
        # H = {H_m}_{m=1}^{N_data}
        channel_matrix = self.channel_model.generate_channel_matrix()
        
        print(f"Channel matrix shape: {channel_matrix.shape}")
        print(f"  (Access Points × Users × Antennas)")
        
        # Train cloud QNN (Algorithm 2)
        cloud_history = self.cloud_qnn.train(
            channel_data=channel_matrix,
            num_iterations=self.network_config.NUM_ITERATIONS_CLOUD
        )
        
        print("\nCloud QNN training completed successfully!")
        return cloud_history
    
    def train_edge_qnns(self, assignment_policy: np.ndarray) -> List[Dict]:
        """
        Train Edge QNNs (Algorithm 1, step 3-8 + Algorithm 3)
        Each edge trains independently with local channel information
        
        Args:
            assignment_policy: Transmitter-user assignment from cloud QNN
        """
        print("\n" + "="*70)
        print("PHASE 2: EDGE QNN TRAINING")
        print("="*70)
        
        edge_histories = []
        
        # Generate channel realizations for edge training
        # Algorithm 1, step 3: for each iteration
        for iteration in range(1):  # Simplified: single iteration
            print(f"\nIteration {iteration + 1}")
            
            # Step 4: Each m-th edge sends channel information to cloud
            # In practice, this is simulated locally
            channel_matrix = self.channel_model.generate_channel_matrix()
            
            # Step 6: Cloud broadcasts assignment γ to all edges
            print(f"Assignment policy shape: {assignment_policy.shape}")
            
            # Step 7: Each m-th edge trains its learning model
            for ap_id in range(self.network_config.NUM_ACCESS_POINTS):
                print(f"\n--- Training Edge QNN for AP {ap_id} ---")
                
                # Extract local channel information H_m for this AP
                local_channel = channel_matrix[ap_id, :, :]
                
                # Train edge QNN (Algorithm 3)
                history = self.edge_qnns[ap_id].train(
                    local_channel=local_channel,
                    assignment=assignment_policy,
                    num_iterations=self.network_config.NUM_ITERATIONS_EDGE
                )
                
                edge_histories.append(history)
        
        print("\n" + "="*70)
        print("All Edge QNNs trained successfully!")
        print("="*70)
        return edge_histories
    
    def deploy(self) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Deployment phase (Algorithm 1, step 9)
        Use trained QNNs for real-time operation
        """
        print("\n" + "="*70)
        print("PHASE 3: DEPLOYMENT")
        print("="*70)
        
        # Acquire instantaneous channel information H
        current_channel = self.channel_model.generate_channel_matrix()
        print("Acquired instantaneous channel information")
        
        # Step 9a: Cloud employs U^cloud to estimate assignment γ
        assignment = self.cloud_qnn.predict(current_channel)
        print(f"\nPredicted assignment policy:")
        print(assignment)
        print(f"Assignment shape: {assignment.shape}")
        
        # Step 9b: Each m-th AP employs U^[m] to estimate precoding v_m
        precoding_vectors = []
        for ap_id in range(self.network_config.NUM_ACCESS_POINTS):
            local_channel = current_channel[ap_id, :, :]
            
            precoding = self.edge_qnns[ap_id].predict(local_channel, assignment)
            precoding_vectors.append(precoding)
            
            print(f"\nAP {ap_id} precoding vector shape: {precoding.shape}")
        
        # Calculate system performance
        performance = self._calculate_system_performance(
            current_channel, assignment, precoding_vectors
        )
        
        print("\n" + "="*70)
        print("DEPLOYMENT RESULTS")
        print("="*70)
        print(f"Total Sum Rate: {performance['sum_rate']:.4f} bits/s/Hz")
        print(f"Average SINR: {performance['avg_sinr']:.4f} dB")
        print(f"Active Users: {performance['active_users']}")
        print("="*70)
        
        return assignment, precoding_vectors, performance
    
    def _calculate_system_performance(self, channel_matrix: np.ndarray,
                                     assignment: np.ndarray,
                                     precoding_vectors: List[np.ndarray]) -> Dict:
        """Calculate overall system performance metrics"""
        sum_rate = 0.0
        sinr_values = []
        active_users = 0
        
        for user_idx in range(self.network_config.NUM_USERS):
            # Find which AP serves this user
            serving_ap = np.argmax(assignment[:, user_idx])
            
            if assignment[serving_ap, user_idx] > 0:
                active_users += 1
                
                # Channel from serving AP to user
                h = channel_matrix[serving_ap, user_idx, :]
                
                # Precoding vector (find user index in precoding matrix)
                user_local_idx = 0  # Simplified indexing
                if precoding_vectors[serving_ap].shape[1] > user_local_idx:
                    v = precoding_vectors[serving_ap][:, user_local_idx]
                    
                    # Signal power
                    signal_power = np.abs(np.dot(h.conj(), v)) ** 2
                    
                    # Noise power (simplified)
                    noise_power = 10 ** (self.network_config.NOISE_POWER_DBM / 10) / 1000
                    
                    # Calculate SINR
                    sinr = signal_power / noise_power
                    sinr_db = 10 * np.log10(sinr + 1e-10)
                    sinr_values.append(sinr_db)
                    
                    # Rate
                    rate = np.log2(1 + sinr)
                    sum_rate += rate
        
        avg_sinr = np.mean(sinr_values) if sinr_values else 0.0
        
        return {
            'sum_rate': sum_rate,
            'avg_sinr': avg_sinr,
            'active_users': active_users,
            'sinr_values': sinr_values
        }
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete multi-stage QNN pipeline
        Implements full Algorithm 1
        """
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + " "*15 + "MULTI-STAGE QNN OPTIMIZATION" + " "*25 + "#")
        print("#" + " "*68 + "#")
        print("#"*70)
        
        # Phase 1: Train Cloud QNN
        cloud_history = self.train_cloud_qnn()
        
        # Get trained assignment policy
        channel_matrix = self.channel_model.generate_channel_matrix()
        assignment_policy = self.cloud_qnn.predict(channel_matrix)
        
        # Phase 2: Train Edge QNNs
        edge_histories = self.train_edge_qnns(assignment_policy)
        
        # Phase 3: Deployment
        assignment, precoding_vectors, performance = self.deploy()
        
        # Compile results
        results = {
            'cloud_history': cloud_history,
            'edge_histories': edge_histories,
            'final_assignment': assignment,
            'final_precoding': precoding_vectors,
            'performance': performance,
            'network_info': self.channel_model.get_network_info()
        }
        
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + " "*20 + "PIPELINE COMPLETED!" + " "*28 + "#")
        print("#" + " "*68 + "#")
        print("#"*70)
        
        return results
    
    def visualize_results(self, results: Dict, save_path: str = None):
        """Visualize the optimization results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Network Topology
        ax = axes[0, 0]
        network_info = results['network_info']
        ap_positions = network_info['ap_positions']
        user_positions = network_info['user_positions']
        
        ax.scatter(ap_positions[:, 0], ap_positions[:, 1], 
                  c='red', s=200, marker='^', label='Access Points', edgecolors='black')
        ax.scatter(user_positions[:, 0], user_positions[:, 1],
                  c='blue', s=100, marker='o', label='Users', edgecolors='black')
        
        # Draw assignment connections
        assignment = results['final_assignment']
        for ap_idx in range(len(ap_positions)):
            for user_idx in range(len(user_positions)):
                if assignment[ap_idx, user_idx] > 0.5:
                    ax.plot([ap_positions[ap_idx, 0], user_positions[user_idx, 0]],
                           [ap_positions[ap_idx, 1], user_positions[user_idx, 1]],
                           'g--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Network Topology and User Assignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cloud QNN Training Loss
        ax = axes[0, 1]
        cloud_losses = results['cloud_history']['losses']
        ax.plot(cloud_losses, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Cloud QNN Training Loss')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Assignment Matrix Heatmap
        ax = axes[1, 0]
        im = ax.imshow(assignment, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('User Index')
        ax.set_ylabel('AP Index')
        ax.set_title('Final Assignment Policy Matrix')
        plt.colorbar(im, ax=ax)
        
        # Plot 4: Performance Metrics
        ax = axes[1, 1]
        metrics = ['Sum Rate\n(bits/s/Hz)', 'Avg SINR\n(dB)', 'Active\nUsers']
        values = [
            results['performance']['sum_rate'],
            results['performance']['avg_sinr'],
            results['performance']['active_users']
        ]
        bars = ax.bar(metrics, values, color=['green', 'orange', 'blue'], alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('System Performance Metrics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")
        
        return fig
