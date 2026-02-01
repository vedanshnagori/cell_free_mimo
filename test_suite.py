"""
Simple Test Script
Verify that all components work correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import NetworkConfig, QNNConfig
from channel_model import WirelessChannel
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN

def test_channel_model():
    """Test channel model generation"""
    print("\n" + "="*60)
    print("TEST 1: Channel Model")
    print("="*60)
    
    config = NetworkConfig()
    channel = WirelessChannel(config)
    
    # Generate channel matrix
    H = channel.generate_channel_matrix()
    
    print(f"✓ Channel matrix shape: {H.shape}")
    print(f"  Expected: ({config.NUM_ACCESS_POINTS}, {config.NUM_USERS}, {config.NUM_ANTENNAS})")
    print(f"✓ Channel matrix dtype: {H.dtype}")
    
    # Get channel features
    features = channel.get_channel_features(H)
    print(f"✓ Channel features shape: {features.shape}")
    print(f"✓ Feature range: [{features.min():.4f}, {features.max():.4f}]")
    
    # Get network info
    info = channel.get_network_info()
    print(f"✓ AP positions shape: {info['ap_positions'].shape}")
    print(f"✓ User positions shape: {info['user_positions'].shape}")
    
    print("✓ Channel model test PASSED")
    return True

def test_cloud_qnn():
    """Test Cloud QNN"""
    print("\n" + "="*60)
    print("TEST 2: Cloud QNN")
    print("="*60)
    
    network_config = NetworkConfig()
    qnn_config = QNNConfig()
    
    # Create Cloud QNN
    cloud_qnn = CloudQNN(
        num_aps=network_config.NUM_ACCESS_POINTS,
        num_users=network_config.NUM_USERS,
        config=qnn_config
    )
    
    print(f"✓ Cloud QNN created with {qnn_config.NUM_QUBITS_CLOUD} qubits")
    print(f"✓ Number of parameters: {cloud_qnn.ansatz.num_parameters}")
    
    # Generate test channel
    channel = WirelessChannel(network_config)
    H = channel.generate_channel_matrix()
    
    # Test encoding
    features = channel.get_channel_features(H)
    encoded = cloud_qnn.encode_channel_info(features)
    print(f"✓ Encoded input shape: {encoded.shape}")
    
    # Test circuit creation
    test_params = np.random.uniform(0, 2*np.pi, cloud_qnn.ansatz.num_parameters)
    qc = cloud_qnn.create_qnn_circuit(encoded, test_params)
    print(f"✓ Quantum circuit created with {qc.num_qubits} qubits")
    print(f"✓ Circuit depth: {qc.depth()}")
    
    # Test short training (2 iterations)
    print("\nRunning short training test (2 iterations)...")
    history = cloud_qnn.train(H, num_iterations=2)
    print(f"✓ Training completed")
    print(f"✓ Final loss: {history['losses'][-1]:.4f}")
    
    # Test prediction
    assignment = cloud_qnn.predict(H)
    print(f"✓ Prediction shape: {assignment.shape}")
    print(f"✓ Assignment sum per user: {assignment.sum(axis=0)}")
    
    print("✓ Cloud QNN test PASSED")
    return True

def test_edge_qnn():
    """Test Edge QNN"""
    print("\n" + "="*60)
    print("TEST 3: Edge QNN")
    print("="*60)
    
    network_config = NetworkConfig()
    qnn_config = QNNConfig()
    
    # Create Edge QNN
    edge_qnn = EdgeQNN(
        ap_id=0,
        num_antennas=network_config.NUM_ANTENNAS,
        config=qnn_config
    )
    
    print(f"✓ Edge QNN created for AP 0 with {qnn_config.NUM_QUBITS_EDGE} qubits")
    print(f"✓ Number of parameters: {edge_qnn.ansatz.num_parameters}")
    
    # Generate test data
    channel = WirelessChannel(network_config)
    H = channel.generate_channel_matrix()
    
    # Create dummy assignment
    assignment = np.zeros((network_config.NUM_ACCESS_POINTS, network_config.NUM_USERS))
    assignment[0, 0] = 1.0  # Assign user 0 to AP 0
    assignment[0, 1] = 1.0  # Assign user 1 to AP 0
    
    local_channel = H[0, :, :]
    
    # Test encoding
    encoded = edge_qnn.encode_local_channel(local_channel, assignment)
    print(f"✓ Encoded local input shape: {encoded.shape}")
    
    # Test circuit creation
    test_params = np.random.uniform(0, 2*np.pi, edge_qnn.ansatz.num_parameters)
    qc = edge_qnn.create_qnn_circuit(encoded, test_params)
    print(f"✓ Quantum circuit created with {qc.num_qubits} qubits")
    print(f"✓ Circuit depth: {qc.depth()}")
    
    # Test short training (2 iterations)
    print("\nRunning short training test (2 iterations)...")
    history = edge_qnn.train(local_channel, assignment, num_iterations=2)
    print(f"✓ Training completed")
    if history['losses']:
        print(f"✓ Final loss: {history['losses'][-1]:.4f}")
    
    # Test prediction
    precoding = edge_qnn.predict(local_channel, assignment)
    print(f"✓ Precoding shape: {precoding.shape}")
    print(f"✓ Precoding dtype: {precoding.dtype}")
    
    print("✓ Edge QNN test PASSED")
    return True

def test_integration():
    """Test integration of all components"""
    print("\n" + "="*60)
    print("TEST 4: Integration Test")
    print("="*60)
    
    from multi_stage_qnn import MultiStageQNN
    
    network_config = NetworkConfig()
    qnn_config = QNNConfig()
    
    # Reduce iterations for quick test
    network_config.NUM_ITERATIONS_CLOUD = 2
    network_config.NUM_ITERATIONS_EDGE = 2
    
    # Create system
    system = MultiStageQNN(network_config, qnn_config)
    print("✓ Multi-stage QNN system created")
    
    # Test cloud training
    print("\nTesting cloud training...")
    cloud_history = system.train_cloud_qnn()
    print(f"✓ Cloud training completed")
    
    # Get assignment
    channel = system.channel_model.generate_channel_matrix()
    assignment = system.cloud_qnn.predict(channel)
    print(f"✓ Assignment predicted: {assignment.shape}")
    
    # Test edge training
    print("\nTesting edge training...")
    edge_histories = system.train_edge_qnns(assignment)
    print(f"✓ Edge training completed for {len(edge_histories)} APs")
    
    # Test deployment
    print("\nTesting deployment...")
    assignment, precoding, performance = system.deploy()
    print(f"✓ Deployment completed")
    print(f"✓ Performance metrics:")
    print(f"  - Sum Rate: {performance['sum_rate']:.4f}")
    print(f"  - Avg SINR: {performance['avg_sinr']:.4f}")
    print(f"  - Active Users: {performance['active_users']}")
    
    print("✓ Integration test PASSED")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "RUNNING TEST SUITE" + " "*25 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    tests = [
        ("Channel Model", test_channel_model),
        ("Cloud QNN", test_cloud_qnn),
        ("Edge QNN", test_edge_qnn),
        ("Integration", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED with error:")
            print(f"  {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*20 + "TEST SUMMARY" + " "*26 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    print("\n" + "="*60)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! 🎉\n")
        return True
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed\n")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
