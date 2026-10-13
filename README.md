# Decoding Electricity Demand: Analyzing Consumption Patterns and Forecasting Future Energy Load

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end academic research and engineering pipeline for multivariate electricity demand analysis, multi-horizon forecasting, and Neural Granger causal discovery using real-world smart grid meter data.

---

## 📌 Project Overview

This repository investigates electricity consumption dynamics using the benchmark **UCI Electricity Load Diagrams (2011–2014)** dataset. The raw dataset records smart meter readings every 15 minutes across **370 distinct industrial and residential clients** over a 4-year period in Portugal.

The project is structured into progressive academic review phases:

### Phase 1: Exploratory Data Analysis & Statistical Foundations (Review 1)
* **Ingestion & Preprocessing:** Custom parsing of 710 MB raw European-formatted tabular logs (`sep=';'`, `decimal=','`).
* **Stationarity Testing:** Mathematical proof of non-stationarity on raw load (ADF p-value $= 0.643$) and strict stationarity after first-order differencing $d=1$ (ADF p-value $= 1.71 \times 10^{-10}$).
* **Classical Decomposition:** Additive series decomposition isolating multi-year trend, 7-day industrial weekend cyclicality, and annual holiday shocks (Christmas & New Year drops).
* **Hyperparameter Identification:** Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis justifying $(p=1, d=1, q=1)$ with seasonal order $m=7$.

### Phase 2: Predictive Modeling, Advanced Deep Learning & Causal Analysis (Review 2)
Benchmarking an evolutionary hierarchy of forecasting paradigms on a 60-day out-of-sample test horizon:
1. **Classical Econometric Baselines:**
   * **ARIMA(1,1,1):** Standard linear autoregressive moving-average model.
   * **SARIMA(1,1,1)x(1,0,1,7):** Seasonal econometric model capturing weekly periodicity ($m=7$).
   * *Analytical Deep Dive:* Mathematical proof of why multi-step open-loop ARIMA/SARIMA forecasts decay into flat straight lines, and demonstration of dynamic 1-step walk-forward Kalman filtering.
2. **Supervised Machine Learning:**
   * **XGBoost Regressor:** Gradient-boosted decision trees over autoregressive lag features ($\text{lag}_1, \text{lag}_7, \text{lag}_{14}$), rolling statistics, and calendar metadata.
3. **Classical Recurrent Deep Learning:**
   * **Vanilla RNN (`nn.RNN`):** 2-layer Elman recurrent network with Reversible Instance Normalization (RevIN).
   * **Standard LSTM (`nn.LSTM`):** 2-layer Long Short-Term Memory network with gated cell state memory.
4. **State-of-the-Art 2026 Hybrid Architectures (Replicated from Literature):**
   * **CLM-Former (*Nature Scientific Reports, 2026*):** Series decomposition + frequency-domain FFT Autocorrelation with the **CLM-subNet** (1D-CNN + bottleneck LSTM) replacing point-wise feed-forward networks.
   * **MKJR (*Elsevier Engineering Applications of Artificial Intelligence, 2026*):** Reversible Instance Normalization (RevIN) + two-stage Kolmogorov-Arnold Network (KAN) learnable B-spline/RBF edge mappings + asymmetric bidirectional dual-stream Mamba (Selective State Spaces) trained with matrix-free random-projection Jacobian regularization.
5. **Neural Granger Causal Discovery:**
   * Evaluation-time input-output Jacobian sensitivity analysis revealing temporal causality pulses at Days $-1, -7, -14$.
6. **Multi-Horizon Stress Testing ($H=14$):**
   * Multi-output rolling forecast evaluation across extended horizons.

---

## 📂 Repository Structure

```text
TimeSeriesAnalysisAndForecasting/
├── data/                                # Dataset storage (ignored in git due to file size)
│   ├── LD2011_2014.txt                  # Raw 710MB 15-minute readings (370 clients)
│   └── daily_aggregated_load.csv        # Pre-aggregated daily Portuguese grid load
├── notebooks/
│   ├── EDA_and_Statistical_Analysis.ipynb          # Review 1: EDA, ADF tests, decomposition, ACF/PACF
│   └── 02_Review2_Modeling_and_Forecasting.ipynb   # Review 2: Complete benchmark, models, & diagnostics
├── Papers/                              # Core 2026 literature PDFs
│   ├── 1-s2.0-S0952197626010845-main.pdf # Paper 1: MKJR Mamba-KAN (Elsevier EAAI 2026)
│   └── 41598_2025_Article_34870.pdf      # Paper 2: CLM-Former (Nature Sci Rep 2026)
├── src/
│   └── generate_modeling_notebook.py    # Automated notebook generator script
├── literature_review.md                 # Master academic literature synthesis
├── Paper_1_MKJR_Mamba_KAN.md            # Detailed architectural breakdown of Paper 1
├── Paper_2_CLM_Former.md                # Detailed architectural breakdown of Paper 2
├── README.md                            # Project landing documentation
└── .gitignore                           # Git exclusion rules for large datasets & venvs
```

---

## 📊 Summary of Model Paradigms (Review 2)

| Model | Paradigm | Strengths | Typical Role |
| :--- | :--- | :--- | :--- |
| **ARIMA(1,1,1)** | Linear Econometric | Simple, mathematically interpretable | Baseline reference; flattens over multi-step horizons |
| **SARIMA(1,1,1)x(1,0,1,7)** | Seasonal Econometric | Captures 7-day cyclical weekend drops | Linear seasonal benchmark |
| **XGBoost** | Gradient Boosted Trees | Exceptional on short-horizon tabular lag features | Strong non-linear reference |
| **Vanilla RNN** | Recurrent Neural Net | Models sequential state transitions | Classical sequential baseline |
| **Standard LSTM** | Gated Recurrent Net | Mitigates vanishing gradients via memory gates | Foundational deep sequential baseline |
| **CLM-Former** | Attention + CNN-LSTM | Cross-domain synergy: FFT autocorrelation + local CNN filtering | State-of-the-art residential load forecasting |
| **MKJR** | State Space + KAN | Linear $O(L)$ Mamba scaling + spline edges + causal attribution | State-of-the-art long-term forecasting & causality |

---

## 🛠️ Quickstart & Reproduction

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Seventie/TimeSeries.git
cd TimeSeries

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install torch pandas numpy statsmodels xgboost scikit-learn matplotlib plotly nbformat nbconvert ipykernel
```

### 2. Dataset Setup
1. Download `electricityloaddiagrams20112014.zip` from the [UCI Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).
2. Extract and place `LD2011_2014.txt` into the `data/` directory.

### 3. Running the Notebooks
Launch Jupyter or open VSCode:
```powershell
jupyter lab
```
* Open [`notebooks/EDA_and_Statistical_Analysis.ipynb`](notebooks/EDA_and_Statistical_Analysis.ipynb) for **Review 1**.
* Open [`notebooks/02_Review2_Modeling_and_Forecasting.ipynb`](notebooks/02_Review2_Modeling_and_Forecasting.ipynb) for **Review 2**. All models and interactive Plotly figures are pre-computed and immediately viewable.

---

## 📜 Citations & Academic References
1. **Zhang et al. (2026)**, *"A unified neural framework for long-term time series forecasting and granger-style causal analysis"*, *Engineering Applications of Artificial Intelligence*, Elsevier, Vol. 176, Art. 114802.
2. **Rahmatinia et al. (2026)**, *"CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model"*, *Scientific Reports*, Nature Portfolio, Vol. 16, Art. 4704.
