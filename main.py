"""
Main Simulation Script
Run the complete Multi-Stage QNN optimization for wireless networks
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import NetworkConfig, QNNConfig
from multi_stage_qnn import MultiStageQNN

def main():
    """Main simulation function"""
    
    print("\n" + "="*70)
    print("  QUANTUM NEURAL NETWORK WIRELESS OPTIMIZATION SIMULATION")
    print("="*70)
    
    # Initialize configurations
    network_config = NetworkConfig()
    qnn_config = QNNConfig()
    
    print("\nSimulation Parameters:")
    print(f"  Network Configuration:")
    print(f"    - Access Points: {network_config.NUM_ACCESS_POINTS}")
    print(f"    - Users: {network_config.NUM_USERS}")
    print(f"    - Antennas per AP: {network_config.NUM_ANTENNAS}")
    print(f"    - Area: {network_config.AREA_WIDTH}m × {network_config.AREA_HEIGHT}m")
    print(f"\n  QNN Configuration:")
    print(f"    - Cloud QNN Qubits: {qnn_config.NUM_QUBITS_CLOUD}")
    print(f"    - Edge QNN Qubits: {qnn_config.NUM_QUBITS_EDGE}")
    print(f"    - QNN Layers: {network_config.NUM_LAYERS}")
    print(f"    - Entanglement: {qnn_config.ENTANGLEMENT}")
    print(f"\n  Training Configuration:")
    print(f"    - Cloud Iterations: {network_config.NUM_ITERATIONS_CLOUD}")
    print(f"    - Edge Iterations: {network_config.NUM_ITERATIONS_EDGE}")
    print(f"    - Learning Rate: {network_config.LEARNING_RATE}")
    print("="*70)
    
    # Create Multi-Stage QNN system
    qnn_system = MultiStageQNN(network_config, qnn_config)
    
    # Run complete pipeline
    results = qnn_system.run_complete_pipeline()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save numerical results
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    
    # Prepare serializable results
    serializable_results = {
        'timestamp': timestamp,
        'network_config': {
            'num_aps': network_config.NUM_ACCESS_POINTS,
            'num_users': network_config.NUM_USERS,
            'num_antennas': network_config.NUM_ANTENNAS,
        },
        'qnn_config': {
            'cloud_qubits': qnn_config.NUM_QUBITS_CLOUD,
            'edge_qubits': qnn_config.NUM_QUBITS_EDGE,
            'layers': network_config.NUM_LAYERS,
        },
        'performance': results['performance'],
        'cloud_training': {
            'final_loss': float(results['cloud_history']['losses'][-1]),
            'final_quality': float(results['cloud_history']['qualities'][-1]),
        },
        'assignment_matrix': results['final_assignment'].tolist(),
    }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        viz_file = os.path.join(results_dir, f"visualization_{timestamp}.png")
        qnn_system.visualize_results(results, save_path=viz_file)
        print(f"Visualization saved to: {viz_file}")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("  SIMULATION SUMMARY")
    print("="*70)
    print(f"\nNetwork Performance:")
    print(f"  - Total Sum Rate: {results['performance']['sum_rate']:.4f} bits/s/Hz")
    print(f"  - Average SINR: {results['performance']['avg_sinr']:.4f} dB")
    print(f"  - Active Users: {results['performance']['active_users']}/{network_config.NUM_USERS}")
    
    print(f"\nTraining Performance:")
    print(f"  - Cloud QNN Final Loss: {results['cloud_history']['losses'][-1]:.4f}")
    print(f"  - Cloud QNN Final Quality: {results['cloud_history']['qualities'][-1]:.4f}")
    
    print(f"\nUser Assignment:")
    for user_idx in range(network_config.NUM_USERS):
        assigned_ap = np.argmax(results['final_assignment'][:, user_idx])
        print(f"  - User {user_idx} → AP {assigned_ap}")
    
    print("="*70)
    print("\nSimulation completed successfully!")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    main()
