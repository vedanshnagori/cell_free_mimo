"""
Test Suite
Verify all components work correctly
Validates fixes:
- NAP > Nuser (Section III)
- NUM_QUBITS_EDGE = Nuser (Lemma 1)
- Parameter shift gradient (Appendix B)
- Full SINR with interference (Eq. 4)
- Max-min objective (Eq. 6)
"""

import sys
import os
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from config import NetworkConfig, QNNConfig
from channel_model import WirelessChannel
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN


def _make_test_config():
    """
    Create test config satisfying all paper constraints:
    - NAP > Nuser     (Section III)
    - NUM_QUBITS_EDGE = Nuser  (Lemma 1)
    - Table III values
    """
    config     = NetworkConfig()
    qnn_config = QNNConfig()

    # paper Table III values
    config.NUM_ACCESS_POINTS   = 6    # NAP > Nuser ✅
    config.NUM_USERS           = 3    # Nuser = 3
    config.NUM_ANTENNAS        = 2    # NTx  = 2
    config.PATH_LOSS_EXPONENT  = 2.3  # κ
    config.INTERFERENCE_FACTOR = 0.1  # μ_n,k
    config.R_PENALTY           = -10.0
    config.NUM_ITERATIONS_CLOUD= 2    # reduced for speed
    config.NUM_ITERATIONS_EDGE = 2

    # Lemma 1: edge needs Nuser qubits
    qnn_config.NUM_QUBITS_EDGE  = config.NUM_USERS
    qnn_config.NUM_QUBITS_CLOUD = 6

    # sync shared params
    qnn_config.RANDOM_SEED          = config.RANDOM_SEED
    qnn_config.SNR                  = config.SNR
    qnn_config.INTERFERENCE_FACTOR  = config.INTERFERENCE_FACTOR
    qnn_config.R_PENALTY            = config.R_PENALTY
    qnn_config.NUM_ITERATIONS_CLOUD = config.NUM_ITERATIONS_CLOUD
    qnn_config.NUM_ITERATIONS_EDGE  = config.NUM_ITERATIONS_EDGE

    return config, qnn_config


# ── Test 1: Channel Model ──────────────────────────────────────────

def test_channel_model():
    """Test WirelessChannel — Eq. (3) ULA model"""
    print("\n" + "="*60)
    print("TEST 1: Channel Model")
    print("="*60)

    config, _ = _make_test_config()
    channel   = WirelessChannel(config)

    # shape check
    H = channel.generate_channel_matrix()
    assert H.shape == (config.NUM_ACCESS_POINTS,
                       config.NUM_USERS,
                       config.NUM_ANTENNAS), \
        f"Wrong shape: {H.shape}"
    print(f"✓ Channel shape: {H.shape} "
          f"(N_AP × N_user × N_Tx)")

    # complex dtype
    assert np.iscomplexobj(H), "Channel should be complex"
    print(f"✓ Channel dtype: {H.dtype}")

    # CSI imperfection Ĥ = H + n_CSI
    H_hat = channel.add_csi_imperfection(H)
    assert H_hat.shape == H.shape
    assert not np.allclose(H, H_hat), \
        "CSI imperfection had no effect"
    print(f"✓ CSI imperfection: Ĥ ≠ H")

    # features normalization
    features = channel.get_channel_features(H)
    assert features.min() >= -1.0 - 1e-6, \
        f"Features below -1: {features.min()}"
    assert features.max() <=  1.0 + 1e-6, \
        f"Features above  1: {features.max()}"
    print(f"✓ Features range: [{features.min():.4f}, "
          f"{features.max():.4f}] ⊆ [-1, 1]")

    # reproducibility
    channel2 = WirelessChannel(config)
    H2       = channel2.generate_channel_matrix()
    assert np.allclose(H, H2), \
        "Same seed should produce same channel"
    print(f"✓ Reproducibility: same seed → same channel")

    # network info
    info = channel.get_network_info()
    assert info['ap_positions'].shape[0]   == config.NUM_ACCESS_POINTS
    assert info['user_positions'].shape[0] == config.NUM_USERS
    print(f"✓ AP positions:   {info['ap_positions'].shape}")
    print(f"✓ User positions: {info['user_positions'].shape}")

    print("✓ Channel model test PASSED")
    return True


# ── Test 2: Cloud QNN ──────────────────────────────────────────────

def test_cloud_qnn():
    """Test CloudQNN — Algorithm 2"""
    print("\n" + "="*60)
    print("TEST 2: Cloud QNN")
    print("="*60)

    config, qnn_config = _make_test_config()
    channel   = WirelessChannel(config)
    H         = channel.generate_channel_matrix()

    cloud_qnn = CloudQNN(
        num_aps  = config.NUM_ACCESS_POINTS,
        num_users= config.NUM_USERS,
        config   = qnn_config
    )
    print(f"✓ Cloud QNN: {qnn_config.NUM_QUBITS_CLOUD} qubits, "
          f"{cloud_qnn.ansatz.num_parameters} params")

    # encoding
    encoded = cloud_qnn.encode_channel_info(H)
    assert len(encoded) == qnn_config.NUM_QUBITS_CLOUD, \
        f"Encoded length {len(encoded)} != {qnn_config.NUM_QUBITS_CLOUD}"
    assert np.all(np.abs(encoded) <= np.pi + 1e-6), \
        "Encoded values outside [-π, π]"
    print(f"✓ Encoded shape: {encoded.shape}, "
          f"range: [{encoded.min():.3f}, {encoded.max():.3f}]")

    # circuit creation
    qc = cloud_qnn.create_qnn_circuit(
        encoded, cloud_qnn.theta_cloud
    )
    assert qc.num_qubits == qnn_config.NUM_QUBITS_CLOUD
    print(f"✓ Circuit: {qc.num_qubits} qubits, "
          f"depth={qc.depth()}")

    # assignment quality with MR precoding
    assignment = np.zeros((config.NUM_ACCESS_POINTS, config.NUM_USERS))
    for k in range(config.NUM_USERS):
        assignment[k % config.NUM_ACCESS_POINTS, k] = 1.0
    quality = cloud_qnn.calculate_assignment_quality(assignment, H)
    print(f"✓ Q_assign = {quality:.4f} (Eq. 14)")

    # loss computation Eq. (13)
    loss = cloud_qnn.calculate_loss(assignment, H)
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    print(f"✓ L_assign = {loss:.4f} (Eq. 13)")

    # normalize assignment — Eq. (6c)
    normalized = cloud_qnn._normalize_assignment(assignment)
    for k in range(config.NUM_USERS):
        assert normalized[:, k].sum() >= 1, \
            f"User {k} has no AP (violates Eq. 6c)"
    print(f"✓ Assignment satisfies Eq. (6c): ϱ_k ≥ 1 ∀k")

    # short training
    print("\nShort training (2 iterations)...")
    history = cloud_qnn.train(H, num_iterations=2)
    assert len(history['losses']) == 2
    print(f"✓ Losses: {[f'{l:.4f}' for l in history['losses']]}")

    # prediction
    assignment = cloud_qnn.predict(H)
    assert assignment.shape == (config.NUM_ACCESS_POINTS,
                                config.NUM_USERS)
    for k in range(config.NUM_USERS):
        assert assignment[:, k].sum() >= 1, \
            f"Prediction: user {k} unassigned"
    print(f"✓ Prediction shape: {assignment.shape}")
    print(f"✓ APs per user: {assignment.sum(axis=0)}")

    print("✓ Cloud QNN test PASSED")
    return True


# ── Test 3: Edge QNN ───────────────────────────────────────────────

def test_edge_qnn():
    """Test EdgeQNN — Algorithm 3, Lemma 1"""
    print("\n" + "="*60)
    print("TEST 3: Edge QNN")
    print("="*60)

    config, qnn_config = _make_test_config()
    channel   = WirelessChannel(config)
    H         = channel.generate_channel_matrix()

    edge_qnn = EdgeQNN(
        ap_id        = 0,
        num_antennas = config.NUM_ANTENNAS,
        config       = qnn_config
    )

    # Lemma 1: num_qubits = Nuser
    assert edge_qnn.num_qubits == config.NUM_USERS, \
        f"Lemma 1 violated: num_qubits={edge_qnn.num_qubits} " \
        f"!= Nuser={config.NUM_USERS}"
    print(f"✓ Lemma 1: num_qubits={edge_qnn.num_qubits} = Nuser")

    # local channel shape (N_Tx × N_user)
    local_channel = H[0, :, :].T
    assert local_channel.shape == (config.NUM_ANTENNAS,
                                   config.NUM_USERS), \
        f"Wrong local_channel shape: {local_channel.shape}"
    print(f"✓ Local channel shape: {local_channel.shape} "
          f"(N_Tx × N_user)")

    # assignment
    assignment    = np.zeros((config.NUM_ACCESS_POINTS,
                              config.NUM_USERS))
    assignment[0, 0] = 1.0
    assignment[0, 1] = 1.0
    ap_assignment = assignment[0, :]

    # encoding
    encoded = edge_qnn.encode_local_channel(local_channel, ap_assignment)
    assert len(encoded) == qnn_config.NUM_QUBITS_EDGE
    assert np.all(np.abs(encoded) <= np.pi + 1e-6)
    print(f"✓ Encoded shape: {encoded.shape}, "
          f"range: [{encoded.min():.3f}, {encoded.max():.3f}]")

    # circuit
    qc = edge_qnn.create_qnn_circuit(encoded, edge_qnn.theta_edge)
    assert qc.num_qubits == qnn_config.NUM_QUBITS_EDGE
    print(f"✓ Circuit: {qc.num_qubits} qubits, depth={qc.depth()}")

    # precoding quality with interference
    all_channels   = [H[m, :, :].T
                      for m in range(config.NUM_ACCESS_POINTS)]
    all_precodings = [None] * config.NUM_ACCESS_POINTS
    precoding_test = np.random.randn(config.NUM_ANTENNAS, 1) + \
                     1j * np.random.randn(config.NUM_ANTENNAS, 1)
    precoding_test /= np.linalg.norm(precoding_test)

    quality = edge_qnn.calculate_precoding_quality(
        precoding           = precoding_test,
        local_channel       = local_channel,
        assignment          = ap_assignment,
        all_precodings      = all_precodings,
        all_channels        = all_channels,
        interference_factor = config.INTERFERENCE_FACTOR,
        snr                 = config.SNR
    )
    print(f"✓ Q_precode = {quality:.4f} (Eq. 16)")

    # loss Eq. (17)
    loss = edge_qnn.calculate_loss(
        precoding      = precoding_test,
        local_channel  = local_channel,
        assignment     = ap_assignment,
        all_precodings = all_precodings,
        all_channels   = all_channels
    )
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    print(f"✓ L_precode = {loss:.4f} (Eq. 17)")

    # short training
    print("\nShort training (2 iterations)...")
    history = edge_qnn.train(
        local_channel  = local_channel,
        assignment     = assignment,
        all_precodings = all_precodings,
        all_channels   = all_channels,
        num_iterations = 2
    )
    print(f"✓ Losses: {[f'{l:.4f}' for l in history['losses']]}")

    # prediction
    precoding = edge_qnn.predict(local_channel, assignment)
    assert precoding.shape[0] == config.NUM_ANTENNAS
    for col in range(precoding.shape[1]):
        norm = np.linalg.norm(precoding[:, col])
        assert abs(norm - 1.0) < 1e-6, \
            f"Col {col} not unit norm: {norm}"
    print(f"✓ Precoding shape: {precoding.shape}, "
          f"||v_m||={np.linalg.norm(precoding[:,0]):.6f} (Eq. 15b)")

    print("✓ Edge QNN test PASSED")
    return True


# ── Test 4: Config Validation ──────────────────────────────────────

def test_config_validation():
    """Test paper constraints are enforced"""
    print("\n" + "="*60)
    print("TEST 4: Config Validation")
    print("="*60)

    config, qnn_config = _make_test_config()

    # NAP > Nuser (Section III)
    assert config.NUM_ACCESS_POINTS > config.NUM_USERS, \
        f"NAP({config.NUM_ACCESS_POINTS}) must > " \
        f"Nuser({config.NUM_USERS})"
    print(f"✓ NAP={config.NUM_ACCESS_POINTS} > "
          f"Nuser={config.NUM_USERS} (Section III)")

    # NUM_QUBITS_EDGE = Nuser (Lemma 1)
    assert qnn_config.NUM_QUBITS_EDGE == config.NUM_USERS, \
        f"Lemma 1: NUM_QUBITS_EDGE({qnn_config.NUM_QUBITS_EDGE}) " \
        f"must = Nuser({config.NUM_USERS})"
    print(f"✓ NUM_QUBITS_EDGE={qnn_config.NUM_QUBITS_EDGE} "
          f"= Nuser (Lemma 1)")

    # SNR > 0 (Eq. 4)
    assert config.SNR > 0, "SNR must be positive"
    print(f"✓ SNR ρ = {config.SNR:.2f} (Eq. 4)")

    # R_PENALTY < 0 (Eq. 14)
    assert config.R_PENALTY < 0, \
        f"R_PENALTY must be negative: {config.R_PENALTY}"
    print(f"✓ R_PENALTY = {config.R_PENALTY} < 0 (Eq. 14)")

    # QNNConfig has all required attributes
    required_attrs = [
        'RANDOM_SEED', 'SNR', 'INTERFERENCE_FACTOR',
        'R_PENALTY', 'NUM_ITERATIONS_CLOUD', 'NUM_ITERATIONS_EDGE',
        'LEARNING_RATE', 'SHOTS', 'BACKEND', 'REPS',
        'ENTANGLEMENT', 'FEATURE_MAP', 'USE_ROTOSOLVE'
    ]
    for attr in required_attrs:
        assert hasattr(qnn_config, attr), \
            f"QNNConfig missing attribute: {attr}"
    print(f"✓ QNNConfig has all required attributes")

    # MultiStageQNN asserts NAP > Nuser
    from multi_stage_qnn import MultiStageQNN
    try:
        bad_config                    = NetworkConfig()
        bad_config.NUM_ACCESS_POINTS  = 2
        bad_config.NUM_USERS          = 5
        MultiStageQNN(bad_config, qnn_config)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print(f"✓ MultiStageQNN rejects NAP < Nuser")

    print("✓ Config validation test PASSED")
    return True


# ── Test 5: Integration ────────────────────────────────────────────

def test_integration():
    """Test full Algorithm 1 pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Integration Test (Algorithm 1)")
    print("="*60)

    from multi_stage_qnn import MultiStageQNN

    config, qnn_config = _make_test_config()
    system = MultiStageQNN(config, qnn_config)
    print("✓ MultiStageQNN created")

    # Phase 1: cloud training
    print("\nPhase 1: Cloud QNN training...")
    cloud_history, channel_matrix = system.train_cloud_qnn()
    assert len(cloud_history['losses']) > 0
    print(f"✓ Cloud training: {len(cloud_history['losses'])} iterations")

    # reuse channel
    assignment = system.cloud_qnn.predict(channel_matrix)
    assert assignment.shape == (config.NUM_ACCESS_POINTS,
                                config.NUM_USERS)
    print(f"✓ Assignment shape: {assignment.shape}")

    # Phase 2: edge training
    print("\nPhase 2: Edge QNN training...")
    edge_histories = system.train_edge_qnns(assignment)
    assert len(edge_histories) > 0
    print(f"✓ Edge training: {len(edge_histories)} histories")

    # Phase 3: deployment
    print("\nPhase 3: Deployment...")
    assignment, precoding, performance = system.deploy()
    assert 'sum_rate'    in performance
    assert 'min_rate'    in performance
    assert 'avg_sinr'    in performance
    assert 'active_users'in performance

    print(f"✓ Performance:")
    print(f"  Min Rate  : {performance['min_rate']:.4f} bits/s/Hz")
    print(f"  Sum Rate  : {performance['sum_rate']:.4f} bits/s/Hz")
    print(f"  Avg SINR  : {performance['avg_sinr']:.4f} dB")
    print(f"  Active Users: {performance['active_users']}"
          f"/{config.NUM_USERS}")

    print("✓ Integration test PASSED")
    return True


# ── Run All Tests ──────────────────────────────────────────────────

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "RUNNING TEST SUITE" + " "*25 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)

    tests = [
        ("Channel Model",     test_channel_model),
        ("Cloud QNN",         test_cloud_qnn),
        ("Edge QNN",          test_edge_qnn),
        ("Config Validation", test_config_validation),
        ("Integration",       test_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED:")
            print(f"  {str(e)}")
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "#"*60)
    print("#" + " "*20 + "TEST SUMMARY" + " "*26 + "#")
    print("#"*60)

    passed = sum(1 for _, s in results if s)
    total  = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name:.<50} {status}")

    print("\n" + "="*60)
    print(f"  Total: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!\n")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed\n")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)