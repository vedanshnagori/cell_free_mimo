# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Qiskit (quantum computing framework)
- NumPy (numerical computations)
- Matplotlib (visualization)
- Other required packages

### Step 2: Verify Installation

Run the test suite to ensure everything is working:

```bash
python test_suite.py
```

Expected output: All 4 tests should pass ✓

### Step 3: Run Your First Simulation

Execute the main simulation:

```bash
python main.py
```

This will:
1. Create a wireless network with 4 APs and 6 users
2. Train the Cloud QNN (50 iterations)
3. Train Edge QNNs for each AP (30 iterations)
4. Deploy and evaluate performance
5. Save results to `results/` directory

**Expected Runtime**: 5-15 minutes (depending on your hardware)

### Step 4: View Results

After completion, check the `results/` directory for:
- `results_*.json` - Numerical results and metrics
- `visualization_*.png` - Network topology and performance plots

## 📊 Understanding the Output

### Console Output

During execution, you'll see:

```
============================================================
PHASE 1: CLOUD QNN TRAINING
============================================================
Training Cloud QNN for Transmitter-User Assignment...
  Iteration 0: Loss = 10.5234, Quality = 5.2341
  Iteration 10: Loss = 8.3421, Quality = 6.1234
  ...

============================================================
PHASE 2: EDGE QNN TRAINING
============================================================
--- Training Edge QNN for AP 0 ---
Training Edge QNN for AP 0...
  Iteration 0: Loss = 3.4561, Quality = 2.1234
  ...

============================================================
PHASE 3: DEPLOYMENT
============================================================
Predicted assignment policy:
[[1. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 1.]
 [0. 0. 1. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0.]]

DEPLOYMENT RESULTS
============================================================
Total Sum Rate: 15.2341 bits/s/Hz
Average SINR: 12.3456 dB
Active Users: 6/6
```

### Results JSON File

```json
{
  "timestamp": "20260201_143022",
  "performance": {
    "sum_rate": 15.234,
    "avg_sinr": 12.345,
    "active_users": 6
  },
  "assignment_matrix": [
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ]
}
```

### Visualization

The visualization shows 4 subplots:

1. **Network Topology**: Shows AP locations (red triangles), user locations (blue circles), and assignment connections (green dashed lines)

2. **Cloud QNN Training Loss**: Shows how the loss decreases during training

3. **Assignment Matrix Heatmap**: Visual representation of which users are assigned to which APs

4. **Performance Metrics**: Bar chart showing sum rate, SINR, and active users

## ⚙️ Quick Configuration Changes

### Change Network Size

Edit `src/config.py`:

```python
# Small network (fast)
NUM_ACCESS_POINTS = 3
NUM_USERS = 4

# Medium network (default)
NUM_ACCESS_POINTS = 4
NUM_USERS = 6

# Large network (slow)
NUM_ACCESS_POINTS = 6
NUM_USERS = 10
```

### Adjust Training Iterations

For faster results (less accurate):
```python
NUM_ITERATIONS_CLOUD = 20  # Default: 50
NUM_ITERATIONS_EDGE = 10   # Default: 30
```

For better results (slower):
```python
NUM_ITERATIONS_CLOUD = 100
NUM_ITERATIONS_EDGE = 50
```

### Change QNN Architecture

```python
# Fewer qubits (faster, less capacity)
NUM_QUBITS_CLOUD = 6
NUM_QUBITS_EDGE = 4

# More qubits (slower, more capacity)
NUM_QUBITS_CLOUD = 10
NUM_QUBITS_EDGE = 8
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'qiskit'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Simulation is too slow
**Solutions**:
1. Reduce number of iterations in `config.py`
2. Reduce number of APs/users
3. Reduce number of qubits
4. Reduce SHOTS in QNNConfig (less accurate measurements)

### Issue: "Memory Error"
**Solutions**:
1. Reduce NUM_QUBITS (each additional qubit doubles memory)
2. Reduce network size
3. Close other applications

### Issue: Poor performance metrics
**Solutions**:
1. Increase training iterations
2. Adjust learning rate (try 0.001 to 0.1)
3. Increase number of qubits
4. Change optimizer (try 'SPSA' instead of 'COBYLA')

## 📚 Next Steps

1. **Experiment with Parameters**: Try different network sizes, qubit numbers, and training iterations

2. **Analyze Results**: Compare performance across different configurations

3. **Modify Algorithms**: The code is modular - you can easily modify individual components

4. **Read the Full README**: Check `README.md` for detailed technical information

5. **Explore the Code**: Each file has detailed comments explaining the implementation

## 💡 Tips for Best Results

1. **Start Small**: Begin with small networks (3 APs, 4 users) to understand behavior

2. **Monitor Convergence**: Check that training loss decreases - if not, adjust learning rate

3. **Compare Configurations**: Run multiple simulations with different settings

4. **Save Results**: Results are timestamped, so you can compare different runs

5. **Visualize**: Always check the visualization to understand network behavior

## 📞 Need Help?

- Check the full `README.md` for technical details
- Review the test suite (`test_suite.py`) for working examples
- Look at inline code comments for algorithm details
- Open an issue if you encounter bugs

---

Happy Simulating! 🎉
