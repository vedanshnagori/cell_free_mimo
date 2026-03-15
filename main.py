"""
Main Simulation Script
Run complete Multi-Stage QNN optimization
Implements Algorithm 1 from:
'Non-Centralized Quantum Neural Networks for Cell-Free MIMO Systems'
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import NetworkConfig, QNNConfig
from multi_stage_qnn import MultiStageQNN


def main():
    """
    Run full Algorithm 1:
    Phase 1: Cloud QNN training (Algorithm 2)
    Phase 2: Edge QNN training  (Algorithm 3)
    Phase 3: Deployment
    """

    print("\n" + "="*70)
    print("  QUANTUM NEURAL NETWORK WIRELESS OPTIMIZATION")
    print("  Non-Centralized QNN for Cell-Free MIMO Systems")
    print("="*70)

    # ── Initialize Configurations ──────────────────────────────────
    network_config = NetworkConfig()
    qnn_config     = QNNConfig()

    print("\nSimulation Parameters:")
    print(f"  Network:")
    print(f"    APs={network_config.NUM_ACCESS_POINTS}, "
          f"Users={network_config.NUM_USERS}, "
          f"Antennas={network_config.NUM_ANTENNAS}")
    print(f"    Area={network_config.AREA_WIDTH}×"
          f"{network_config.AREA_HEIGHT}m, "
          f"κ={network_config.PATH_LOSS_EXPONENT}")
    print(f"    SNR={network_config.SNR:.2f}, "
          f"μ_nk={network_config.INTERFERENCE_FACTOR}")

    print(f"\n  QNN:")
    print(f"    Cloud qubits={qnn_config.NUM_QUBITS_CLOUD}, "
          f"Edge qubits={qnn_config.NUM_QUBITS_EDGE}")
    print(f"    Layers={qnn_config.NUM_LAYERS}, "      # ✅ from QNNConfig
          f"REPS={qnn_config.REPS}, "
          f"Entanglement={qnn_config.ENTANGLEMENT}")
    print(f"    Optimizer={qnn_config.OPTIMIZER}, "
          f"SHOTS={qnn_config.SHOTS}")

    print(f"\n  Training (Table III):")
    print(f"    Cloud iters={network_config.NUM_ITERATIONS_CLOUD}, "
          f"Edge iters={network_config.NUM_ITERATIONS_EDGE}")
    print(f"    LR={network_config.LEARNING_RATE}, "
          f"r_penalty={network_config.R_PENALTY}")
    print(f"    N_data={network_config.N_DATA}, "
          f"N_epoch={network_config.N_EPOCH}")
    print(f"    Seed={network_config.RANDOM_SEED}")
    print("="*70)

    # ── Run Pipeline ───────────────────────────────────────────────
    qnn_system = MultiStageQNN(network_config, qnn_config)
    results    = qnn_system.run_complete_pipeline()

    # ── Save Results ───────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # edge final losses
    edge_final_losses = [
        float(h['losses'][-1])
        for h in results['edge_histories']
        if len(h['losses']) > 0
    ]

    serializable_results = {
        'timestamp'   : timestamp,
        'random_seed' : network_config.RANDOM_SEED,   # reproducibility

        'network_config': {
            'num_aps'             : network_config.NUM_ACCESS_POINTS,
            'num_users'           : network_config.NUM_USERS,
            'num_antennas'        : network_config.NUM_ANTENNAS,
            'path_loss_exponent'  : network_config.PATH_LOSS_EXPONENT,
            'snr'                 : network_config.SNR,
            'interference_factor' : network_config.INTERFERENCE_FACTOR,
        },

        'qnn_config': {
            'cloud_qubits' : qnn_config.NUM_QUBITS_CLOUD,
            'edge_qubits'  : qnn_config.NUM_QUBITS_EDGE,
            'layers'       : qnn_config.NUM_LAYERS,     # ✅ QNNConfig
            'reps'         : qnn_config.REPS,
            'shots'        : qnn_config.SHOTS,
        },

        # key paper metrics (Eq. 6 max-min objective)
        'performance': {
            'sum_rate'    : results['performance']['sum_rate'],
            'min_rate'    : results['performance']['min_rate'],  # ✅ added
            'avg_sinr'    : results['performance']['avg_sinr'],
            'active_users': results['performance']['active_users'],
            'user_rates'  : results['performance'].get('user_rates', []),
            'sinr_values' : results['performance'].get('sinr_values', [])
        },

        'cloud_training': {
            'final_loss'   : float(results['cloud_history']['losses'][-1]),
            'final_quality': float(results['cloud_history']['qualities'][-1]),
            'all_losses'   : [float(l)
                              for l in results['cloud_history']['losses']]
        },

        # edge training — Fig. 6
        'edge_training': {
            'num_edges'    : len(results['edge_histories']),
            'final_losses' : edge_final_losses,
            'avg_final_loss': float(np.mean(edge_final_losses))
                              if edge_final_losses else 0.0
        },

        'assignment_matrix': results['final_assignment'].tolist()
    }

    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # ── Visualization ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        viz_file = os.path.join(
            results_dir, f"visualization_{timestamp}.png"
        )
        qnn_system.visualize_results(results, save_path=viz_file)
        print(f"Visualization saved to: {viz_file}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SIMULATION SUMMARY")
    print("="*70)

    print(f"\nNetwork Performance (Eq. 6 max-min objective):")
    print(f"  Min Rate  : "               # primary paper metric
          f"{results['performance']['min_rate']:.4f} bits/s/Hz")
    print(f"  Sum Rate  : "
          f"{results['performance']['sum_rate']:.4f} bits/s/Hz")
    print(f"  Avg SINR  : "
          f"{results['performance']['avg_sinr']:.4f} dB")
    print(f"  Active Users: "
          f"{results['performance']['active_users']}/"
          f"{network_config.NUM_USERS}")

    print(f"\nTraining Performance:")
    print(f"  Cloud QNN Final Loss:     "
          f"{results['cloud_history']['losses'][-1]:.4f}")    # Fig. 5
    print(f"  Cloud QNN Final Quality:  "
          f"{results['cloud_history']['qualities'][-1]:.4f}")
    if edge_final_losses:
        print(f"  Edge QNN Avg Final Loss:  "
              f"{np.mean(edge_final_losses):.4f}")            # Fig. 6

    # ── User Assignment (many-to-one per Section IV-A) ─────────────
    print(f"\nUser Assignment (γ):")
    for k in range(network_config.NUM_USERS):
        assigned_aps = np.where(
            results['final_assignment'][:, k] > 0.5
        )[0]
        rate = (results['performance']['user_rates'][k]
                if k < len(results['performance'].get('user_rates', []))
                else 0.0)
        print(f"  User {k} → APs {assigned_aps.tolist():20s} "
              f"Rate={rate:.4f} bits/s/Hz")

    print("="*70)
    print("Simulation completed!")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
