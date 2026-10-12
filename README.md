<div align="center">

# 🏗️ Physics-Informed Structural Analysis Engine

### Non-linear Moment–Curvature (M–φ) Prediction for Concrete-Filled Steel Tube (CFST) Columns

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

*Structural Mechanics × Deep Learning*

</div>


## 📖 Overview

This repository implements a **physics-informed deep learning surrogate model** that simulates the non-linear Moment–Curvature (**M–φ**) response of Concrete-Filled Steel Tube (CFST) columns in milliseconds — replacing the computational cost of traditional Finite Element Method (FEM) software while staying strictly consistent with structural mechanics.

A custom Python **Fiber-Section Analysis** engine, grounded in Euler–Bernoulli beam theory, generates ground-truth physics. A **Dual-Output Seq2Seq LSTM** is then trained on top of it, simultaneously predicting structural capacity (Moment) *and* the internal deformation state (Neutral Axis depth) — enforcing strain compatibility and giving the model a degree of interpretability that standard black-box regressors don't have.

> **In short:** a fast neural surrogate that doesn't just predict an output — it predicts the internal physics that explains it.

---

## 🔑 Key Contributions

**🧮 Custom Structural Solver**
A Python-based Fiber-Section Analysis solver using Brent's root-finding algorithm (`scipy.optimize.brentq`) to simulate dynamic Moment–Curvature hysteresis for CFST sections — discretizing the cross-section into 100 fiber layers, with bi-linear steel plasticity and non-linear concrete confinement/softening behavior.

**📐 Eurocode-Compliant Probabilistic Data Sampler**
A parallelized data generator constrained by **Eurocode 2 (EN 1992-1-1)** and **Eurocode 4 (EN 1994-1-1)**:

| Parameter | Symbol | Range |
|---|---|---|
| Outer diameter | D | 200 – 800 mm |
| Wall thickness | t | 4 – 20 mm |
| Steel yield strength | f<sub>y</sub> | 235 – 460 MPa |
| Concrete compressive strength | f<sub>c</sub> | 30 – 80 MPa |
| Axial load (multi-storey) | N | 1,000 – 15,000 kN |

Strict slenderness (width-to-thickness) checks reject any configuration prone to premature local buckling.

**⚡ Multi-Core Parallel Data Generation**
Uses `joblib` (`Parallel(n_jobs=-1)`) to generate 2,000 samples of 100-step non-linear structural response data across all available CPU cores.

**🔬 Glass-Box Dual-Output LSTM**
A Seq2Seq network mapping 4 static design parameters **[D, t, f<sub>y</sub>, f<sub>c</sub>]** → a 100-step time series **[M(φ), c(φ)]**, predicting the internal latent physical variable (Neutral Axis depth) *alongside* Moment capacity — proving internal force equilibrium rather than treating it as a black box.

**📈 Comprehensive Model Benchmarking**
Automated benchmarking and Taylor Diagram evaluation across 9 ML models (XGBoost, LightGBM, SVR, Random Forest, and more) to identify the most accurate and stable surrogate architecture.

---

## 🧠 Network Architecture

```text
Input  →  [D, t, fy, fc]                         (4 static design parameters)
            │
            ▼
      RepeatVector (sequence length = 100)
            │
            ▼
      LSTM  →  128 units · tanh · return_sequences
            │
            ▼
      Dropout (0.1)
            │
            ▼
      LSTM  →  64 units · tanh · return_sequences
            │
            ▼
      Dropout (0.1)
            │
            ▼
      TimeDistributed(Dense(2))                   →  [M(φ), c(φ)]
```

<div align="center">

**Total Parameters: 117,634** · all trainable

</div>

---

## 📊 Results

### Multi-Task Target Generation

The Neutral Axis dynamically shifts to maintain internal equilibrium as flexural capacity plateaus due to material non-linearity (steel yielding, concrete plastic softening/crushing).

<img width="1009" alt="Scaled Moment and Neutral Axis targets" src="https://github.com/user-attachments/assets/cc878d54-5238-4795-b8fb-5273f6ed0d48" />

<sub>**Left** — Scaled Moment capacity M · **Right** — Shifting Neutral Axis depth c, maintaining physical equilibrium</sub>

### Training Convergence & Surrogate Accuracy

| Metric | Result |
|---|---|
| Training epochs | 150 (EarlyStopping + ReduceLROnPlateau) |
| Validation MSE | ≈ 3.5 × 10⁻⁴ |
| Convergence shape | Clean L-curve, no overfitting |
| Surrogate vs. ground truth | Closely tracks physics-engine trajectories on held-out test samples |

<img width="1589" alt="Convergence curve and surrogate vs ground truth" src="https://github.com/user-attachments/assets/9276f07f-f4f2-4ba6-b6cc-f5bac68c8b40" />

<sub>**Left** — MSE loss curve (log scale) · **Right** — Surrogate Moment prediction vs. physics-engine ground truth, multiple test samples</sub>

---

## 📁 File Structure

<details>
<summary>Click to expand</summary>

```text
.
├── notebooks/
│   ├── cfst_structural_engine.ipynb    # Fiber-section solver, data generation, LSTM training
│   └── model_comparison.ipynb          # Benchmarking across 9 ML models + Taylor diagram
├── src/
│   ├── generate_data.py                # Standalone: Eurocode-constrained physics data generator
│   ├── train_lstm.py                   # Standalone: trains the Dual-Output Glass-Box LSTM
│   └── benchmark_models.py             # Standalone: 9-model benchmark + Taylor diagram
├── assets/
│   ├── multitask_targets.png           # Training targets plot (Moment + Neutral Axis)
│   └── surrogate_and_convergence.png   # Convergence curve & surrogate accuracy plot
├── requirements.txt                    # Environment dependencies
├── LICENSE                             # MIT License
└── README.md
```

</details>

---

## ⚙️ Installation

<details>
<summary>Click to expand</summary>

```bash
git clone https://github.com/kavyamtomriddle/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
```

**Core dependencies**

- `numpy`, `pandas`, `matplotlib`
- `scipy` — Brent's root-finding algorithm
- `scikit-learn`, `xgboost`, `lightgbm`
- `tensorflow` / `keras` — Dual-Output Seq2Seq LSTM
- `joblib` — parallel Monte Carlo data generation

</details>

---

## 🚀 Usage

**1 · Data Generation** — generate Eurocode-constrained CFST samples via Fiber-Section Analysis

```bash
python src/generate_data.py
```

**2 · Model Benchmarking** — compare tree-based and scale-sensitive algorithms via the automated Taylor Diagram script

```bash
python src/benchmark_models.py
```

**3 · Train the Dual-Output LSTM** — train the explainable surrogate model

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

Contributions, issues, and feature requests are welcome — check the [issues page](https://github.com/kavyamtomriddle/<repository-name>/issues).

## 📄 License

Licensed under the [MIT License](LICENSE).
