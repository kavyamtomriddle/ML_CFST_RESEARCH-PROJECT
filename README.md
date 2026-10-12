# Physics-Informed Structural Analysis Engine 🏗️🤖

### Concrete-Filled Steel Tube (CFST) Columns: Non-linear Moment-Curvature (M-φ) Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-green.svg)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Tech Stack:** Python, SciPy, TensorFlow, Fiber-Section Analysis, FEM · Structural Mechanics & AI

## 📖 Project Overview

This repository implements a **physics-informed deep learning surrogate model** that simulates the non-linear Moment-Curvature ($M$-$\phi$) behavior of Concrete-Filled Steel Tube (CFST) columns in milliseconds — bypassing the computational cost of traditional Finite Element Method (FEM) software while staying strictly consistent with structural mechanics.

To achieve this, I built a custom Python **Fiber-Section Analysis** engine grounded in Euler-Bernoulli beam theory, then trained a **Dual-Output Seq2Seq LSTM** that simultaneously predicts structural capacity (Moment) *and* the internal deformation state (Neutral Axis depth) — enforcing mathematical strain compatibility and giving the model a degree of explainability that standard black-box regressors lack.

## Key Contributions

* **Custom Structural Solver** — A Python-based Fiber-Section Analysis solver using Brent's root-finding algorithm (`scipy.optimize.brentq`) to simulate dynamic Moment-Curvature hysteresis for CFST sections, discretizing the cross-section into 100 fiber layers and modeling bi-linear steel plasticity alongside non-linear concrete confinement/softening behavior.
* **Eurocode-Compliant Probabilistic Data Sampler** — A parallelized data generator constrained by **Eurocode 2 (EN 1992-1-1)** and **Eurocode 4 (EN 1994-1-1)**, sampling outer diameter D ∈ [200, 800] mm, wall thickness t ∈ [4, 20] mm, steel yield strength fy ∈ [235, 460] MPa, concrete strength fc ∈ [30, 80] MPa, and realistic multi-storey axial loads (1,000–15,000 kN) — with strict slenderness (width-to-thickness) checks to reject configurations prone to premature local buckling.
* **Multi-Core Parallel Data Generation Pipeline** — Uses `joblib` (`Parallel(n_jobs=-1)`) to generate 2,000 samples of 100-step non-linear structural response data across all available CPU cores.
* **Glass-Box Dual-Output LSTM Architecture** — A Seq2Seq network mapping 4 static design parameters [D, t, fy, fc] to a 100-step time series [M(φ), c(φ)], predicting the internal latent physical variable (Neutral Axis depth) alongside Moment capacity — proving internal force equilibrium rather than treating it as a black box.
* **Comprehensive Model Benchmarking** — Automated benchmarking and Taylor Diagram evaluation across 9 ML models (including XGBoost, LightGBM, SVR, and Random Forest) to identify the most accurate and stable surrogate architecture.

---

## 🧠 Network Architecture

```text
Input (4 static params: D, t, fy, fc)
   │
   ▼
RepeatVector (Sequence Length = 100)
   │
   ▼
LSTM Layer 1 (128 units, return_sequences=True, activation='tanh')
   │
   ▼
Dropout (rate = 0.1)
   │
   ▼
LSTM Layer 2 (64 units, return_sequences=True, activation='tanh')
   │
   ▼
Dropout (rate = 0.1)
   │
   ▼
TimeDistributed Dense (2 units: [Moment Capacity, Neutral Axis Depth])
```

*Total Parameters: 117,634 (all trainable)*

---

## 📊 Visualizations & Accuracy Verification

### 1. Multi-Task Target Generation (Physical Validity)

Unlike black-box models, this engine's targets expose both the structural capacity *and* the underlying mechanics. The plots below show how the Neutral Axis dynamically shifts to maintain internal equilibrium as flexural capacity plateaus due to material non-linearity (steel yielding and concrete plastic softening/crushing).

<img width="1009" height="470" alt="image" src="https://github.com/user-attachments/assets/cc878d54-5238-4795-b8fb-5273f6ed0d48" />
*Left: Scaled Moment Capacity M. Right: Shifting Neutral Axis Depth c maintaining physical equilibrium.*

### 2. Training Convergence & Surrogate Model Accuracy

* **Convergence Check:** Across 150 training epochs with `EarlyStopping` and `ReduceLROnPlateau`, the validation MSE loss converges in a clean L-shape to ≈ 3.5 × 10⁻⁴ with no overfitting.
* **Surrogate vs. Physics Engine (Ground Truth):** LSTM predictions (dashed) closely reconstruct the non-linear trajectories computed by the physical Fiber-Section solver (solid) on held-out test samples.

<img width="1589" height="590" alt="image" src="https://github.com/user-attachments/assets/9276f07f-f4f2-4ba6-b6cc-f5bac68c8b40" />

*Left: MSE loss learning curve (log scale). Right: Surrogate model Moment prediction vs. physics engine ground truth across multiple test samples.*

---

## 📁 File Structure

```text
.
├── notebooks/
│   ├── cfst_structural_engine.ipynb    # Fiber-section solver, data generation, LSTM training
│   └── model_comparison.ipynb          # Benchmarking across 9 ML models + Taylor diagram
├── assets/
│   ├── multitask_targets.png           # Training targets plot (Moment + Neutral Axis)
│   └── surrogate_and_convergence.png   # Convergence curve & surrogate accuracy plot
├── requirements.txt                    # Environment dependencies
└── README.md
```

---

## ⚙️ Installation & Requirements

```bash
git clone https://github.com/kavyamtomriddle/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
```

**Core Dependencies:**

* `numpy`, `pandas`, `matplotlib`
* `scipy` (Brent's root-finding algorithm)
* `scikit-learn`, `xgboost`, `lightgbm`
* `tensorflow` / `keras` (Dual-Output Seq2Seq LSTM)
* `joblib` (parallel Monte Carlo data generation)

---

## 🚀 Usage

### 1. Data Generation

Run the parallelized sampler to generate Eurocode-constrained CFST samples via Fiber-Section Analysis:

```bash
python src/generate_data.py
```

### 2. Model Benchmarking

Compare tree-based and scale-sensitive algorithms via the automated Taylor Diagram script:

```bash
python src/benchmark_models.py
```

### 3. Train the Dual-Output LSTM

Train the explainable surrogate model to predict capacity and physical state simultaneously:

```bash
python src/train_lstm.py
```

Or simply open `notebooks/cfst_structural_engine.ipynb` in Google Colab or a local Jupyter environment and run the cells sequentially.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Numerical Computation & Optimization | SciPy (`brentq`), NumPy |
| Parallel Processing | Joblib (`Parallel`, `delayed`) |
| Deep Learning | TensorFlow / Keras (Seq2Seq LSTM, TimeDistributed Dense) |
| Preprocessing & Scaling | Scikit-Learn (`MinMaxScaler`, `train_test_split`) |
| Benchmarking | XGBoost, LightGBM, Random Forest, SVR |
| Visualization | Matplotlib |
| Structural Codes | Eurocode 2 (EN 1992-1-1), Eurocode 4 (EN 1994-1-1) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/kavyamtomriddle/<repository-name>/issues).

## 📄 License

This project is licensed under the MIT License.
