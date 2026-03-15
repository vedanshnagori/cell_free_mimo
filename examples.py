"""
Example: Custom Scenario
Demonstrates custom wireless network scenarios
All scenarios respect paper constraints:
- NAP > Nuser (Section III)
- NUM_QUBITS_EDGE = Nuser (Lemma 1)
- Max-min rate objective (Eq. 6)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import NetworkConfig, QNNConfig
from multi_stage_qnn import MultiStageQNN


def _print_results(name: str, results: dict):
    """Print results with min_rate as primary metric (Eq. 6)"""
    perf = results['performance']
    print(f"\n{name} Results:")
    print(f"  Min Rate  : {perf['min_rate']:.4f} bits/s/Hz"
          f"  ← max-min objective (Eq. 6)")
    print(f"  Sum Rate  : {perf['sum_rate']:.4f} bits/s/Hz")
    print(f"  Avg SINR  : {perf['avg_sinr']:.4f} dB")
    print(f"  Active Users: {perf['active_users']}")


def custom_scenario_small_network():
    """
    Scenario 1: Small network — quick testing
    NAP=6 > Nuser=3 (Section III)
    NUM_QUBITS_EDGE = Nuser = 3 (Lemma 1)
    """
    print("\n" + "="*70)
    print("  SCENARIO 1: Small Network (NAP=6, Nuser=3)")
    print("="*70)

    config = NetworkConfig()
    config.NUM_ACCESS_POINTS   = 6    # ✅ NAP > Nuser
    config.NUM_USERS           = 3    # ✅ Nuser = 3 (Table III)
    config.NUM_ANTENNAS        = 2    # ✅ NTx = 2  (Table III)
    config.NUM_ITERATIONS_CLOUD= 20
    config.NUM_ITERATIONS_EDGE = 15
    config.INTERFERENCE_FACTOR = 0.1  # μ_n,k Table III
    config.PATH_LOSS_EXPONENT  = 2.3  # κ Table III

    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_CLOUD = 6
    qnn_config.NUM_QUBITS_EDGE  = config.NUM_USERS  # ✅ Lemma 1: = Nuser

    system  = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    _print_results("Small Network", results)
    return results


def custom_scenario_dense_network():
    """
    Scenario 2: Dense network — more APs and users
    NAP=12 > Nuser=6 (Section III)
    NUM_QUBITS_EDGE = Nuser = 6 (Lemma 1)
    """
    print("\n" + "="*70)
    print("  SCENARIO 2: Dense Network (NAP=12, Nuser=6)")
    print("="*70)

    config = NetworkConfig()
    config.NUM_ACCESS_POINTS   = 12   # ✅ NAP > Nuser
    config.NUM_USERS           = 6
    config.NUM_ANTENNAS        = 2
    config.NUM_ITERATIONS_CLOUD= 30
    config.NUM_ITERATIONS_EDGE = 20
    config.INTERFERENCE_FACTOR = 0.1
    config.PATH_LOSS_EXPONENT  = 2.3

    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_CLOUD = 8
    qnn_config.NUM_QUBITS_EDGE  = config.NUM_USERS  # ✅ Lemma 1: = Nuser

    system  = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    _print_results("Dense Network", results)
    return results


def custom_scenario_massive_mimo():
    """
    Scenario 3: More antennas per AP
    NAP=8 > Nuser=4 (Section III)
    NUM_QUBITS_EDGE = Nuser = 4 (Lemma 1, NOT num_antennas)
    """
    print("\n" + "="*70)
    print("  SCENARIO 3: More Antennas (NAP=8, Nuser=4, NTx=8)")
    print("="*70)

    config = NetworkConfig()
    config.NUM_ACCESS_POINTS   = 8    # ✅ NAP > Nuser
    config.NUM_USERS           = 4
    config.NUM_ANTENNAS        = 8    # more antennas per AP
    config.NUM_ITERATIONS_CLOUD= 40
    config.NUM_ITERATIONS_EDGE = 25
    config.INTERFERENCE_FACTOR = 0.1
    config.PATH_LOSS_EXPONENT  = 2.3

    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_CLOUD = 8
    qnn_config.NUM_QUBITS_EDGE  = config.NUM_USERS  # ✅ Lemma 1: Nuser NOT NTx

    system  = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    _print_results("More Antennas", results)
    return results


def custom_scenario_high_interference():
    """
    Scenario 4: High interference — matches paper Fig. 4 μ_d=0.6
    Tests QNN robustness under heavy inter-AP interference
    NAP=6 > Nuser=3 (Section III)
    """
    print("\n" + "="*70)
    print("  SCENARIO 4: High Interference (μ_d=0.6, Fig. 4)")
    print("="*70)

    config = NetworkConfig()
    config.NUM_ACCESS_POINTS   = 6
    config.NUM_USERS           = 3
    config.NUM_ANTENNAS        = 2
    config.NUM_ITERATIONS_CLOUD= 50
    config.NUM_ITERATIONS_EDGE = 30
    config.INTERFERENCE_FACTOR = 0.6   # μ_d=0.6 from Fig. 4
    config.PATH_LOSS_EXPONENT  = 2.3

    qnn_config = QNNConfig()
    qnn_config.NUM_QUBITS_CLOUD = 6
    qnn_config.NUM_QUBITS_EDGE  = config.NUM_USERS  # ✅ Lemma 1

    system  = MultiStageQNN(config, qnn_config)
    results = system.run_complete_pipeline()
    _print_results("High Interference", results)
    return results


def compare_scenarios():
    """
    Compare all scenarios
    Primary metric: min_rate (max-min objective Eq. 6)
    """
    print("\n" + "#"*70)
    print("#" + " "*15 + "COMPARING SCENARIOS" + " "*35 + "#")
    print("#"*70)

    scenarios = [
        ("Small Network",      custom_scenario_small_network),
        ("Dense Network",      custom_scenario_dense_network),
        ("More Antennas",      custom_scenario_massive_mimo),
        ("High Interference",  custom_scenario_high_interference),
    ]

    results_comparison = []

    for name, scenario_func in scenarios:
        try:
            result = scenario_func()
            results_comparison.append({
                'name'        : name,
                'min_rate'    : result['performance']['min_rate'],  # ✅
                'sum_rate'    : result['performance']['sum_rate'],
                'avg_sinr'    : result['performance']['avg_sinr'],
                'active_users': result['performance']['active_users']
            })
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Comparison Table ───────────────────────────────────────────
    print("\n" + "="*75)
    print("  SCENARIO COMPARISON (sorted by Min Rate — Eq. 6 objective)")
    print("="*75)
    print(f"{'Scenario':<22} {'Min Rate':<12} {'Sum Rate':<12} "
          f"{'Avg SINR':<12} {'Users'}")
    print("-"*75)

    # sort by min_rate — primary paper metric
    results_comparison.sort(key=lambda x: x['min_rate'], reverse=True)

    for r in results_comparison:
        print(f"{r['name']:<22} {r['min_rate']:<12.4f} "
              f"{r['sum_rate']:<12.4f} "
              f"{r['avg_sinr']:<12.4f} "
              f"{r['active_users']}")
    print("="*75)

    return results_comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run custom QNN scenarios'
    )
    parser.add_argument(
        '--scenario', type=str, default='small',
        choices=['small', 'dense', 'mimo', 'interference', 'compare'],
        help='Scenario to run'
    )
    args = parser.parse_args()

    scenario_map = {
        'small'       : custom_scenario_small_network,
        'dense'       : custom_scenario_dense_network,
        'mimo'        : custom_scenario_massive_mimo,
        'interference': custom_scenario_high_interference,
        'compare'     : compare_scenarios
    }

    scenario_map[args.scenario]()
    print("\n✓ Scenario completed!\n")
