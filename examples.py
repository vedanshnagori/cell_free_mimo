"""
Example: Custom Scenario
Demonstrates how to create and run a custom wireless network scenario
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import NetworkConfig, QNNConfig
from multi_stage_qnn import MultiStageQNN

def custom_scenario_small_network():
    """
    Example 1: Small network for quick testing
    3 APs, 4 users, reduced training iterations
    """
    print("\n" + "="*70)
    print("  CUSTOM SCENARIO 1: Small Network")
    print("="*70)
    
    # Create custom configuration
    config = NetworkConfig()
    config.NUM_ACCESS_POINTS = 3
    config.NUM_USERS = 4
    config.NUM_ANTENNAS = 4
    config.NUM_ITERATIONS_CLOUD = 20  # Reduced for speed
    config.NUM_ITERATIONS_EDGE = 15
    
    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_CLOUD = 6
    qnn_config.NUM_QUBITS_EDGE = 4
    
    # Run simulation
    system = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    
    print("\nSmall Network Results:")
    print(f"  Sum Rate: {results['performance']['sum_rate']:.4f}")
    print(f"  Avg SINR: {results['performance']['avg_sinr']:.4f}")
    
    return results

def custom_scenario_dense_network():
    """
    Example 2: Dense network with more users
    4 APs, 10 users
    """
    print("\n" + "="*70)
    print("  CUSTOM SCENARIO 2: Dense Network")
    print("="*70)
    
    config = NetworkConfig()
    config.NUM_ACCESS_POINTS = 4
    config.NUM_USERS = 10
    config.NUM_ANTENNAS = 4
    config.NUM_ITERATIONS_CLOUD = 30
    config.NUM_ITERATIONS_EDGE = 20
    
    qnn_config = QNNConfig()
    
    system = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    
    print("\nDense Network Results:")
    print(f"  Sum Rate: {results['performance']['sum_rate']:.4f}")
    print(f"  Active Users: {results['performance']['active_users']}/10")
    
    return results

def custom_scenario_massive_mimo():
    """
    Example 3: Massive MIMO scenario
    4 APs, 6 users, 8 antennas per AP
    """
    print("\n" + "="*70)
    print("  CUSTOM SCENARIO 3: Massive MIMO")
    print("="*70)
    
    config = NetworkConfig()
    config.NUM_ACCESS_POINTS = 4
    config.NUM_USERS = 6
    config.NUM_ANTENNAS = 8  # More antennas
    config.NUM_ITERATIONS_CLOUD = 40
    config.NUM_ITERATIONS_EDGE = 25
    
    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_EDGE = 8  # More qubits for more antennas
    
    system = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    
    print("\nMassive MIMO Results:")
    print(f"  Sum Rate: {results['performance']['sum_rate']:.4f}")
    print(f"  Avg SINR: {results['performance']['avg_sinr']:.4f}")
    
    return results

def compare_scenarios():
    """Compare different scenarios"""
    print("\n" + "#"*70)
    print("#" + " "*15 + "COMPARING DIFFERENT SCENARIOS" + " "*25 + "#")
    print("#"*70)
    
    scenarios = [
        ("Small Network", custom_scenario_small_network),
        ("Dense Network", custom_scenario_dense_network),
        ("Massive MIMO", custom_scenario_massive_mimo),
    ]
    
    results_comparison = []
    
    for name, scenario_func in scenarios:
        try:
            result = scenario_func()
            results_comparison.append({
                'name': name,
                'sum_rate': result['performance']['sum_rate'],
                'avg_sinr': result['performance']['avg_sinr'],
                'active_users': result['performance']['active_users']
            })
        except Exception as e:
            print(f"\nError in {name}: {e}")
    
    # Print comparison
    print("\n" + "="*70)
    print("  SCENARIO COMPARISON")
    print("="*70)
    print(f"{'Scenario':<20} {'Sum Rate':<15} {'Avg SINR (dB)':<15} {'Users'}")
    print("-"*70)
    
    for result in results_comparison:
        print(f"{result['name']:<20} {result['sum_rate']:<15.4f} "
              f"{result['avg_sinr']:<15.4f} {result['active_users']}")
    
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run custom QNN scenarios')
    parser.add_argument('--scenario', type=str, default='small',
                       choices=['small', 'dense', 'mimo', 'compare'],
                       help='Scenario to run')
    
    args = parser.parse_args()
    
    if args.scenario == 'small':
        custom_scenario_small_network()
    elif args.scenario == 'dense':
        custom_scenario_dense_network()
    elif args.scenario == 'mimo':
        custom_scenario_massive_mimo()
    elif args.scenario == 'compare':
        compare_scenarios()
    
    print("\n✓ Custom scenario completed!\n")
