"""
Multi-Stage QNN System - Algorithm 1
Orchestrates cloud and edge QNNs for cell-free MIMO optimization
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
    Multi-Stage QNN for Cell-Free MIMO Optimization
    Implements Algorithm 1 from paper
    """

    def __init__(self, network_config: NetworkConfig,
                 qnn_config: QNNConfig):
        self.network_config = network_config
        self.qnn_config     = qnn_config

        # channel model — Eq. (3)
        self.channel_model = WirelessChannel(network_config)

        # Cloud QNN — Algorithm 2
        self.cloud_qnn = CloudQNN(
            num_aps  = network_config.NUM_ACCESS_POINTS,
            num_users= network_config.NUM_USERS,
            config   = qnn_config
        )

        # Edge QNNs — Algorithm 3, one per AP
        self.edge_qnns = [
            EdgeQNN(
                ap_id        = ap_id,
                num_antennas = network_config.NUM_ANTENNAS,
                config       = qnn_config
            )
            for ap_id in range(network_config.NUM_ACCESS_POINTS)
        ]

        print(f"Multi-Stage QNN initialized:")
        print(f"  APs={network_config.NUM_ACCESS_POINTS}, "
              f"Users={network_config.NUM_USERS}, "
              f"Antennas={network_config.NUM_ANTENNAS}")
        print(f"  Cloud QNN: {qnn_config.NUM_QUBITS_CLOUD} qubits")
        print(f"  Edge QNN:  {qnn_config.NUM_QUBITS_EDGE} qubits per AP")

        # validate NAP > Nuser (Section III requirement)
        assert network_config.NUM_ACCESS_POINTS > network_config.NUM_USERS, \
            f"NAP ({network_config.NUM_ACCESS_POINTS}) must be > " \
            f"Nuser ({network_config.NUM_USERS}) per paper Section III"

    # ── Phase 1: Cloud QNN Training ────────────────────────────────

    def train_cloud_qnn(self) -> Tuple[Dict, np.ndarray]:
        """
        Algorithm 1 Step 2 + Algorithm 2
        Train cloud QNN with complete network channel info Ĥ
        Returns history and channel used for training
        """
        print("\n" + "="*70)
        print("PHASE 1: CLOUD QNN TRAINING")
        print("="*70)

        # generate Ĥ = {Ĥ_m}^N_AP — Algorithm 1 step 2
        channel_matrix = self.channel_model.generate_channel_matrix()
        print(f"Channel matrix shape: {channel_matrix.shape} "
              f"(N_AP × N_user × N_Tx)")

        # Algorithm 2
        cloud_history = self.cloud_qnn.train(
            channel_data   = channel_matrix,
            num_iterations = self.network_config.NUM_ITERATIONS_CLOUD
        )

        print("Cloud QNN training completed!")
        return cloud_history, channel_matrix

    # ── Phase 2: Edge QNN Training ─────────────────────────────────

    def train_edge_qnns(self,
                        assignment_policy: np.ndarray) -> List[Dict]:
        """
        Algorithm 1 Steps 3-8 + Algorithm 3
        Train each edge QNN with local channel info and cloud assignment γ

        Args:
            assignment_policy: γ from cloud QNN ∈ {0,1}^(N_AP × N_user)
        """
        print("\n" + "="*70)
        print("PHASE 2: EDGE QNN TRAINING")
        print("="*70)

        edge_histories = []

        # Algorithm 1 Step 3: for each t ∈ {1,...,N_iteration}
        for iteration in range(self.network_config.NUM_ITERATIONS_EDGE):
            print(f"\nIteration {iteration + 1}/"
                  f"{self.network_config.NUM_ITERATIONS_EDGE}")

            # Step 4: each AP sends Ĥ_m to cloud
            channel_matrix = self.channel_model.generate_channel_matrix()

            # Step 5: cloud estimates γ with optimized θ^cloud
            assignment_policy = self.cloud_qnn.predict(channel_matrix)

            # Step 6: cloud broadcasts γ to all edges
            print(f"Assignment policy shape: {assignment_policy.shape}")

            # collect all local channels for interference calculation
            all_local_channels = [
                channel_matrix[m, :, :].T   # (N_Tx × N_user)
                for m in range(self.network_config.NUM_ACCESS_POINTS)
            ]

            # initialize precodings for interference (None before training)
            current_precodings = [None] * \
                                  self.network_config.NUM_ACCESS_POINTS

            # Step 7: each m-th edge trains U^[m] (Algorithm 3)
            for ap_id in range(self.network_config.NUM_ACCESS_POINTS):
                print(f"\n--- Training Edge QNN for AP {ap_id} ---")

                # local channel h_m: shape (N_Tx × N_user)
                local_channel = channel_matrix[ap_id, :, :].T

                history = self.edge_qnns[ap_id].train(
                    local_channel  = local_channel,
                    assignment     = assignment_policy,
                    all_precodings = current_precodings,
                    all_channels   = all_local_channels
                )
                edge_histories.append(history)

                # update precodings for subsequent APs
                current_precodings[ap_id] = \
                    self.edge_qnns[ap_id].predict(
                        local_channel, assignment_policy
                    )

        print("\n" + "="*70)
        print("All Edge QNNs trained!")
        print("="*70)
        return edge_histories

    # ── Phase 3: Deployment ────────────────────────────────────────

    def deploy(self) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Algorithm 1 Step 9 — Deployment phase
        Use trained U^cloud and U^[m] for real-time inference
        """
        print("\n" + "="*70)
        print("PHASE 3: DEPLOYMENT")
        print("="*70)

        # acquire instantaneous Ĥ
        current_channel = self.channel_model.generate_channel_matrix()

        # Step 9a: cloud estimates γ with trained U^cloud(θ^cloud)
        assignment = self.cloud_qnn.predict(current_channel)
        print(f"Assignment policy:\n{assignment}")

        # collect all channels for interference
        all_channels = [
            current_channel[m, :, :].T
            for m in range(self.network_config.NUM_ACCESS_POINTS)
        ]

        # Step 9b: each AP estimates v_m with trained U^[m](θ^[m])
        precoding_vectors = [None] * self.network_config.NUM_ACCESS_POINTS

        for ap_id in range(self.network_config.NUM_ACCESS_POINTS):
            local_channel = current_channel[ap_id, :, :].T

            precoding_vectors[ap_id] = self.edge_qnns[ap_id].predict(
                local_channel = local_channel,
                assignment    = assignment
            )
            print(f"AP {ap_id} precoding shape: "
                  f"{precoding_vectors[ap_id].shape}")

        # evaluate performance
        performance = self._calculate_system_performance(
            current_channel, assignment, precoding_vectors
        )

        print("\n" + "="*70)
        print("DEPLOYMENT RESULTS")
        print("="*70)
        print(f"Sum Rate:     {performance['sum_rate']:.4f} bits/s/Hz")
        print(f"Min Rate:     {performance['min_rate']:.4f} bits/s/Hz")
        print(f"Avg SINR:     {performance['avg_sinr']:.4f} dB")
        print(f"Active Users: {performance['active_users']}")
        print("="*70)

        return assignment, precoding_vectors, performance

    # ── Performance Evaluation ─────────────────────────────────────

    def _calculate_system_performance(self,
                                       channel_matrix: np.ndarray,
                                       assignment: np.ndarray,
                                       precoding_vectors: List[np.ndarray]
                                       ) -> Dict:
        """
        Compute sum rate and min rate per Eq. (5)
        Includes inter-AP interference term
        """
        user_rates   = []
        sinr_values  = []
        active_users = 0

        for k in range(self.network_config.NUM_USERS):

            # all APs serving user k (many-to-one)
            serving_aps = np.where(assignment[:, k] > 0.5)[0]
            if len(serving_aps) == 0:
                continue

            active_users += 1

            # ── Signal: Σ_{m∈A_k} ρ|h^T_mk v_m|²  (Eq. 5) ────────
            signal = 0.0
            for m in serving_aps:
                h_mk = channel_matrix[m, k, :]
                if (precoding_vectors[m] is not None and
                        precoding_vectors[m].shape[1] > 0):
                    v_m   = precoding_vectors[m][:, 0]
                    min_l = min(len(h_mk), len(v_m))
                    signal += (self.network_config.SNR *
                               np.abs(h_mk[:min_l].conj() @
                                      v_m[:min_l]) ** 2)

            # ── Interference: ρ Σ_{n∉A_k} μ|h^T_nk v_n|²  (Eq. 5) ─
            interference = 0.0
            for n in range(self.network_config.NUM_ACCESS_POINTS):
                if (assignment[n, k] < 0.5 and
                        precoding_vectors[n] is not None):
                    h_nk  = channel_matrix[n, k, :]
                    v_n   = precoding_vectors[n][:, 0]
                    min_l = min(len(h_nk), len(v_n))
                    interference += (
                        self.network_config.INTERFERENCE_FACTOR *
                        self.network_config.SNR *
                        np.abs(h_nk[:min_l].conj() @
                               v_n[:min_l]) ** 2
                    )

            # ── SINR and Rate  (Eq. 5) ─────────────────────────────
            noise  = self.network_config.NOISE_POWER_LINEAR
            sinr   = signal / (interference + noise)
            rate   = np.log2(1 + sinr)

            user_rates.append(rate)
            sinr_values.append(10 * np.log10(sinr + 1e-10))

        return {
            'sum_rate'   : float(np.sum(user_rates)),
            'min_rate'   : float(np.min(user_rates))
                           if user_rates else 0.0,   # max-min objective
            'avg_sinr'   : float(np.mean(sinr_values))
                           if sinr_values else 0.0,
            'active_users': active_users,
            'user_rates' : user_rates,
            'sinr_values': sinr_values
        }

    # ── Full Pipeline ──────────────────────────────────────────────

    def run_complete_pipeline(self) -> Dict:
        """
        Full Algorithm 1 execution:
        Phase 1: Cloud QNN training
        Phase 2: Edge QNN training
        Phase 3: Deployment
        """
        print("\n" + "#"*70)
        print("#" + " "*15 + "MULTI-STAGE QNN OPTIMIZATION" + " "*25 + "#")
        print("#"*70)

        # Phase 1: train cloud QNN + reuse channel
        cloud_history, channel_matrix = self.train_cloud_qnn()

        # get assignment from trained cloud — reuse same channel
        assignment_policy = self.cloud_qnn.predict(channel_matrix)

        # Phase 2: train edge QNNs
        edge_histories = self.train_edge_qnns(assignment_policy)

        # Phase 3: deploy
        assignment, precoding_vectors, performance = self.deploy()

        results = {
            'cloud_history'   : cloud_history,
            'edge_histories'  : edge_histories,
            'final_assignment': assignment,
            'final_precoding' : precoding_vectors,
            'performance'     : performance,
            'network_info'    : self.channel_model.get_network_info()
        }

        print("\n" + "#"*70)
        print("#" + " "*20 + "PIPELINE COMPLETED!" + " "*28 + "#")
        print("#"*70)
        return results

    # ── Visualization ──────────────────────────────────────────────

    def visualize_results(self, results: Dict,
                          save_path: str = None):
        """
        Reproduce paper figures:
        Fig. 4: Sum rate, Fig. 5: Cloud loss, Fig. 6: Edge loss
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        network_info   = results['network_info']
        ap_positions   = network_info['ap_positions']
        user_positions = network_info['user_positions']
        assignment     = results['final_assignment']

        # ── Plot 1: Network Topology ───────────────────────────────
        ax = axes[0, 0]
        ax.scatter(ap_positions[:, 0], ap_positions[:, 1],
                   c='red', s=200, marker='^',
                   label='APs', edgecolors='black')
        ax.scatter(user_positions[:, 0], user_positions[:, 1],
                   c='blue', s=100, marker='o',
                   label='Users', edgecolors='black')
        for m in range(len(ap_positions)):
            for k in range(len(user_positions)):
                if assignment[m, k] > 0.5:
                    ax.plot(
                        [ap_positions[m,0], user_positions[k,0]],
                        [ap_positions[m,1], user_positions[k,1]],
                        'g--', alpha=0.5, linewidth=2
                    )
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Network Topology and Assignment')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Plot 2: Cloud QNN Loss (Fig. 5) ───────────────────────
        ax = axes[0, 1]
        ax.plot(results['cloud_history']['losses'],
                'b-', linewidth=2, label='Cloud QNN')
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Loss $L_{assign}$')
        ax.set_title('Cloud QNN Training Loss (Fig. 5)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Plot 3: Edge QNN Loss (Fig. 6) ────────────────────────
        ax = axes[0, 2]
        valid = [h for h in results['edge_histories']
                 if len(h['losses']) > 0]
        if valid:
            max_len = max(len(h['losses']) for h in valid)
            avg_losses = [
                np.mean([h['losses'][i] for h in valid
                         if len(h['losses']) > i])
                for i in range(max_len)
            ]
            ax.plot(avg_losses, 'r-', linewidth=2,
                    label='Edge QNN (avg)')
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Loss $L_{precode}$')
        ax.set_title('Edge QNN Training Loss (Fig. 6)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Plot 4: Assignment Matrix Heatmap ──────────────────────
        ax = axes[1, 0]
        im = ax.imshow(assignment, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('User Index')
        ax.set_ylabel('AP Index')
        ax.set_title('Assignment Policy γ')
        plt.colorbar(im, ax=ax)

        # ── Plot 5: User Rates (max-min objective Fig. 4) ─────────
        ax = axes[1, 1]
        user_rates = results['performance'].get('user_rates', [])
        if user_rates:
            ax.bar(range(len(user_rates)), user_rates,
                   color='steelblue', alpha=0.7)
            ax.axhline(y=min(user_rates), color='red',
                       linestyle='--', label=f"Min rate={min(user_rates):.2f}")
        ax.set_xlabel('User Index')
        ax.set_ylabel('Rate (bits/s/Hz)')
        ax.set_title('Per-User Rates (Max-Min Objective)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Plot 6: Performance Summary ────────────────────────────
        ax = axes[1, 2]
        metrics = ['Sum Rate\n(bits/s/Hz)', 'Min Rate\n(bits/s/Hz)',
                   'Avg SINR\n(dB)', 'Active\nUsers']
        values  = [
            results['performance']['sum_rate'],
            results['performance']['min_rate'],
            results['performance']['avg_sinr'],
            results['performance']['active_users']
        ]
        bars = ax.bar(metrics, values,
                      color=['green','red','orange','blue'],
                      alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.2f}', ha='center', va='bottom')
        ax.set_title('System Performance Metrics')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        return fig
