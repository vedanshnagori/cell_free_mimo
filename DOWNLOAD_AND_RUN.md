# 📥 Download and Run Instructions

## Complete Step-by-Step Guide for Running and Uploading to GitHub

---

## 🎯 What You Have

You have a complete Multi-Stage QNN wireless optimization simulator with:
- ✅ Full implementation of 3 algorithms from the research paper
- ✅ Qiskit quantum circuit simulation
- ✅ Realistic wireless channel modeling
- ✅ Automated testing and visualization
- ✅ Ready to run on your computer
- ✅ Ready to upload to GitHub

---

## 📦 PART 1: Download the Project

You should have received a folder called **`qnn_wireless_optimization`** containing these files:

```
qnn_wireless_optimization/
├── run.sh                    # 🚀 Quick run script (Mac/Linux)
├── run.bat                   # 🚀 Quick run script (Windows)
├── main.py                   # Main simulation
├── test_suite.py            # Test all components
├── examples.py              # Custom scenarios
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Full documentation
├── QUICKSTART.md          # Quick start guide
├── PROJECT_SUMMARY.md     # Project overview
├── GITHUB_SETUP.md        # GitHub upload guide
├── DOWNLOAD_AND_RUN.md    # This file
│
└── src/                    # Source code folder
    ├── config.py
    ├── channel_model.py
    ├── cloud_qnn.py
    ├── edge_qnn.py
    └── multi_stage_qnn.py
```

**Save this entire folder somewhere on your computer** (e.g., Desktop, Documents, etc.)

---

## 💻 PART 2: Run Locally (First Time Setup)

### A. Prerequisites

Make sure you have:
1. **Python 3.8 or higher** installed
   - Check: Open terminal/command prompt and type `python --version` or `python3 --version`
   - If not installed: Download from https://www.python.org/downloads/

2. **pip** (Python package manager) - usually comes with Python
   - Check: Type `pip --version` or `pip3 --version`

### B. Quick Run (Easiest Method)

**For Windows Users:**
1. Double-click `run.bat`
2. Choose option 1 to run tests
3. Choose option 2 to run simulation

**For Mac/Linux Users:**
1. Open Terminal
2. Navigate to folder: `cd /path/to/qnn_wireless_optimization`
3. Run: `./run.sh`
4. Choose option 1 to run tests
5. Choose option 2 to run simulation

### C. Manual Setup (If Quick Run Doesn't Work)

**Step 1: Open Terminal/Command Prompt**

Windows: Press `Win + R`, type `cmd`, press Enter
Mac: Press `Cmd + Space`, type "terminal", press Enter
Linux: Press `Ctrl + Alt + T`

**Step 2: Navigate to Project Folder**

```bash
# Replace with your actual path
cd /path/to/qnn_wireless_optimization
```

**Step 3: Create Virtual Environment (Recommended)**

```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On Mac/Linux:
python3 -m venv venv
source venv/bin/activate
```

**Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

This will install:
- Qiskit (quantum computing framework)
- NumPy (numerical computations)
- Matplotlib (visualization)
- Other required packages

**Expected time: 2-5 minutes**

**Step 5: Run Tests**

```bash
python test_suite.py
```

**Expected output:**
```
============================================================
                  RUNNING TEST SUITE
============================================================

============================================================
TEST 1: Channel Model
============================================================
✓ Channel model test PASSED

============================================================
TEST 2: Cloud QNN
============================================================
✓ Cloud QNN test PASSED

============================================================
TEST 3: Edge QNN
============================================================
✓ Edge QNN test PASSED

============================================================
TEST 4: Integration Test
============================================================
✓ Integration test PASSED

============================================================
                    TEST SUMMARY
============================================================
  Channel Model...................................... ✓ PASSED
  Cloud QNN.......................................... ✓ PASSED
  Edge QNN........................................... ✓ PASSED
  Integration........................................ ✓ PASSED

  Total: 4/4 tests passed

  🎉 ALL TESTS PASSED! 🎉
```

**If all tests pass, you're ready to run simulations!**

**Step 6: Run Main Simulation**

```bash
python main.py
```

**What happens:**
- Initializes 4 Access Points and 6 Users
- Trains Cloud QNN (50 iterations) - ~3-5 minutes
- Trains Edge QNNs (30 iterations per AP) - ~5-10 minutes
- Deploys and evaluates performance
- Saves results to `results/` folder

**Expected time: 10-15 minutes total**

**Step 7: Check Results**

Look in the `results/` folder for:
- `results_YYYYMMDD_HHMMSS.json` - Performance data
- `visualization_YYYYMMDD_HHMMSS.png` - Network plots

---

## 🎮 PART 3: Run Different Scenarios

Try different network configurations:

```bash
# Small network (faster, ~5 minutes)
python examples.py --scenario small

# Dense network with more users (~15 minutes)
python examples.py --scenario dense

# Massive MIMO with more antennas (~20 minutes)
python examples.py --scenario mimo

# Compare all scenarios (~30-40 minutes)
python examples.py --scenario compare
```

---

## 🚀 PART 4: Upload to GitHub

### Option A: GitHub Desktop (Easiest - No Command Line)

**Step 1: Install GitHub Desktop**
- Download: https://desktop.github.com/
- Install and sign in with your GitHub account

**Step 2: Create Repository**
1. Open GitHub Desktop
2. Click **File → New Repository**
3. Repository name: `qnn-wireless-optimization`
4. Local path: Select the **parent folder** containing `qnn_wireless_optimization`
5. **Uncheck** "Initialize with README" (we already have one)
6. Click **Create Repository**

**Step 3: Add Files**
1. GitHub Desktop shows all your files
2. In the bottom-left, enter commit message: `Initial commit: Multi-stage QNN implementation`
3. Click **Commit to main**

**Step 4: Publish to GitHub**
1. Click **Publish repository** (top bar)
2. Choose **Public** or **Private**
3. Click **Publish Repository**

**✅ Done! Your code is on GitHub!**

Your repository URL: `https://github.com/YOUR_USERNAME/qnn-wireless-optimization`

---

### Option B: Command Line (More Control)

**Step 1: Install Git**
- Windows: https://git-scm.com/download/win
- Mac: Already installed or run `xcode-select --install`
- Linux: `sudo apt install git` or `sudo yum install git`

**Step 2: Configure Git (First Time Only)**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Step 3: Create Repository on GitHub Website**
1. Go to https://github.com/new
2. Repository name: `qnn-wireless-optimization`
3. Description: "Multi-stage Quantum Neural Network for wireless network optimization"
4. Choose **Public** or **Private**
5. **Do NOT** check "Initialize with README"
6. Click **Create repository**

**Step 4: Upload Your Code**

In your terminal, navigate to the project folder:

```bash
cd /path/to/qnn_wireless_optimization

# Initialize git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Multi-stage QNN implementation"

# Connect to GitHub (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/qnn-wireless-optimization.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Enter your GitHub username and password (or token) when prompted.**

**✅ Done! Your code is on GitHub!**

---

## 🎨 PART 5: Make Your Repository Look Professional

### Add Topics/Tags on GitHub
1. Go to your repository on GitHub
2. Click the ⚙️ icon next to "About"
3. Add topics: `quantum-computing`, `qiskit`, `wireless-networks`, `optimization`, `machine-learning`
4. Save changes

### Add Repository Description
In the same "About" section:
- Description: "Multi-stage Quantum Neural Network (QNN) for joint transmitter-user assignment and transmit precoding optimization in wireless networks using Qiskit"

### Update Repository (After Making Changes)

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with message
git commit -m "Description of what you changed"

# Push to GitHub
git push
```

---

## 📊 Understanding the Output

### Console Output During Simulation

```
======================================================================
PHASE 1: CLOUD QNN TRAINING
======================================================================
Training Cloud QNN for Transmitter-User Assignment...
  Iteration 0: Loss = 10.5234, Quality = 5.2341
  Iteration 10: Loss = 8.3421, Quality = 6.1234
  Iteration 20: Loss = 6.2341, Quality = 7.5432
  ...

======================================================================
PHASE 2: EDGE QNN TRAINING
======================================================================
Training Edge QNN for AP 0...
  Iteration 0: Loss = 3.4561, Quality = 2.1234
  ...

======================================================================
PHASE 3: DEPLOYMENT
======================================================================
Predicted assignment policy:
[[1. 0. 0. 0. 1. 0.]     ← Users 0 and 4 assigned to AP 0
 [0. 1. 0. 0. 0. 1.]     ← Users 1 and 5 assigned to AP 1
 [0. 0. 1. 1. 0. 0.]     ← Users 2 and 3 assigned to AP 2
 [0. 0. 0. 0. 0. 0.]]    ← No users assigned to AP 3

======================================================================
DEPLOYMENT RESULTS
======================================================================
Total Sum Rate: 15.2341 bits/s/Hz
Average SINR: 12.3456 dB
Active Users: 6/6
======================================================================
```

### Results Files

**JSON File (`results_*.json`):**
```json
{
  "timestamp": "20260201_143022",
  "performance": {
    "sum_rate": 15.234,        // Total data rate
    "avg_sinr": 12.345,        // Signal quality
    "active_users": 6          // Successfully assigned users
  },
  "cloud_training": {
    "final_loss": 2.341,
    "final_quality": 8.567
  },
  "assignment_matrix": [...]   // User-to-AP mapping
}
```

**Visualization PNG:**
- Top-left: Network topology with connections
- Top-right: Training loss curve
- Bottom-left: Assignment heatmap
- Bottom-right: Performance metrics

---

## ⚙️ Customization Guide

### Change Network Size

Edit `src/config.py`:

```python
class NetworkConfig:
    NUM_ACCESS_POINTS = 4      # Change this
    NUM_USERS = 6              # Change this
    NUM_ANTENNAS = 4           # Change this
```

### Adjust Training Speed

For faster results (less accurate):
```python
NUM_ITERATIONS_CLOUD = 20      # Default: 50
NUM_ITERATIONS_EDGE = 10       # Default: 30
```

For better results (slower):
```python
NUM_ITERATIONS_CLOUD = 100
NUM_ITERATIONS_EDGE = 50
```

---

## 🐛 Troubleshooting

### Problem: "python: command not found"
**Solution:** Use `python3` instead of `python`

### Problem: "pip: command not found"
**Solution:** Use `pip3` instead of `pip`, or install pip

### Problem: Installation fails with permission error
**Solution:** 
- Use virtual environment (recommended)
- Or use `pip install --user -r requirements.txt`

### Problem: Simulation is too slow
**Solutions:**
1. Reduce iterations in `src/config.py`
2. Use smaller network (fewer APs/users)
3. Reduce number of qubits

### Problem: "git: command not found"
**Solution:** Install Git from https://git-scm.com/

### Problem: GitHub authentication fails
**Solutions:**
1. Use GitHub Desktop (no authentication needed)
2. Use personal access token instead of password
3. Guide: https://docs.github.com/en/authentication

---

## ✅ Final Checklist

Before sharing your GitHub repository:

- [ ] All tests pass (`python test_suite.py`)
- [ ] Main simulation runs successfully
- [ ] Results are generated correctly
- [ ] .gitignore file is in place
- [ ] README is clear
- [ ] Repository has description and topics on GitHub
- [ ] No sensitive information in code

---

## 📚 Documentation Files Reference

- **README.md** - Complete technical documentation
- **QUICKSTART.md** - 5-minute getting started guide
- **PROJECT_SUMMARY.md** - Project overview and features
- **GITHUB_SETUP.md** - Detailed GitHub instructions
- **DOWNLOAD_AND_RUN.md** - This file

---

## 🎓 What's Next?

1. **Experiment** with different configurations
2. **Analyze** results in the `results/` folder
3. **Modify** algorithms in the `src/` folder
4. **Share** your GitHub repository
5. **Collaborate** with others

---

## 📞 Need Help?

**Common Resources:**
- Python installation: https://www.python.org/
- Git installation: https://git-scm.com/
- GitHub guides: https://guides.github.com/
- Qiskit documentation: https://qiskit.org/documentation/

**Check Documentation:**
- Look at README.md for technical details
- Check inline code comments
- Review example scenarios in examples.py

---

## 🎉 Success!

If you can:
- ✅ Run `python test_suite.py` successfully
- ✅ Run `python main.py` and get results
- ✅ See your repository on GitHub

**You're all set!** 🚀

Your repository URL: `https://github.com/YOUR_USERNAME/qnn-wireless-optimization`

---

**Last Updated:** February 2026
**Version:** 1.0.0
