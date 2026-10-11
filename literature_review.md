# Synthesis of Recent Advances in Multi-Horizon Time Series Forecasting and Causal Modeling

This document provides an in-depth academic synthesis of the two primary research papers provided in `Papers/`:
1. **Paper 1 (EAAI 2026):** *"A unified neural framework for long-term time series forecasting and granger-style causal analysis"* (MKJR Framework).
2. **Paper 2 (Nature Sci Rep 2026):** *"CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model"* (CLM-Former Framework).

Both papers investigate long-term multivariate forecasting on the **UCI Electricity dataset** (321 clients), making them directly applicable to our project.

---

## 1. Comparative Overview of the Two Papers

| Dimension | Paper 1: MKJR (Elsevier EAAI 2026) | Paper 2: CLM-Former (Nature Sci Rep 2026) |
| :--- | :--- | :--- |
| **Primary Domain** | Long-Term Forecasting + Granger Causal Discovery | Smart Grid Multi-Horizon Electricity Demand Prediction |
| **Core Architecture** | Mamba State Space Model (SSM) + Kolmogorov-Arnold Network (KAN) | Autoformer Transformer Backbone + CLM-subNet |
| **Sequence Complexity** | Linear $O(L)$ via Selective State Spaces | Sub-quadratic $O(L \log L)$ via FFT Autocorrelation |
| **Nonlinear Mapping** | Edge-learnable B-spline functions (KAN layers) | 1D-CNN + Bottleneck LSTM sub-network |
| **Decomposition Strategy** | Variable-wise dual-stream gating (Detail vs. Global context) | Layer-wise Additive Decomposition (Trend vs. Seasonal) |
| **Regularization Scheme** | Matrix-free Random-Projection Jacobian Regularization | Spline/recurrent dropout and early stopping |
| **Interpretability Output** | Lag-resolved Granger causal graph via Jacobian sensitivity | Autocorrelation lag maps and gradient saliency maps |
| **UCI Electricity MSE** | **0.162** (Multi-horizon avg) | **0.209** (Horizons: 96, 192, 336) |

---

## 2. In-Depth Breakdown: Paper 1 (MKJR)

### The Problem It Solves
Traditional forecasting architectures (like vanilla Transformers, DLinear, and LSTMs) prioritize pure numerical prediction, acting as "black boxes" that cannot explain *how* or *why* different variables influence each other over time. Conversely, classical econometric causal tools (such as Vector Autoregression and Granger causality) are strictly linear and cannot scale to high dimensions or long sequences.

### Exact Architectural Design:
1. **Reversible Instance Normalization (RevIN):** Normalizes inputs dynamically per window to prevent distribution shifts, removing non-stationarity before feeding to layers.
2. **KAN Encoder-Decoder:**
   * Instead of standard MLPs with fixed activations ($\text{ReLU}(Wx)$), KANs place **learnable B-splines directly on the network edges**.
   * The first KAN encoder extracts an **encoder-side source-lag structural prior** ($\pi_{i,\alpha}$), mathematically determining which historical lag $\alpha$ of variable $i$ retains predictive signal.
3. **Asymmetric Bidirectional Dual-Stream Mamba Backbone:**
   * Uses Mamba State-Space Models (SSMs) to achieve $O(L)$ linear scaling.
   * **Detail Stream:** Processes high-dimensional latent tokens modulated by local temporal signatures (extracted with a kernel size 3 1D-CNN) *without* residual connections to retain sharp local shocks.
   * **Global Stream:** Processes compressed tokens modulated by global channel pooling *with* a residual connection to preserve coarse multi-variable context.
   * Employs bidirectional forward and backward scanning for comprehensive temporal context.
4. **Matrix-Free Random-Projection Jacobian Regularization:**
   * Penalizes excessive model sensitivity without computing the full $O(M^2 L T)$ dense Jacobian matrix.
   * Uses Hutchinson-style random probes ($v \sim \mathcal{N}(0, I)$ and Rademacher $p \sim \pm 1$) to regularize directional gradients:
     $$\mathcal{L} = \text{MSE} + \lambda_{sens} R_{sens} + \lambda_{str} R_{str}$$
5. **Granger Causal Graph Recovery:**
   * At inference time, combines the KAN structural prior ($\pi_{i,\alpha}$) with the absolute input-output Jacobian sensitivity to produce continuous causal scores.
   * Runs circular permutation surrogate tests with Benjamini-Hochberg False Discovery Rate (FDR) control to extract an interpretable causal adjacency matrix.

---

## 3. In-Depth Breakdown: Paper 2 (CLM-Former)

### The Problem It Solves
State-of-the-art Transformers (such as Autoformer) excel at detecting broad periodic cycles via frequency-domain autocorrelation. However, they route the seasonal component through point-wise Feed-Forward Networks (FFNs). Point-wise FFNs compute transformations independently at each timestamp, which completely destroys local temporal dependencies and prevents the model from predicting abrupt, aperiodic load surges (e.g., unexpected factory spikes or heatwave peaks).

### Exact Architectural Design:
1. **Multi-Scale Additive Series Decomposition:**
   * At every layer of both the encoder and decoder, the series is decomposed:
     $$X_t = \text{AvgPooling}(\text{Padding}(X)), \quad X_s = X - X_t$$
   * The **Trend component ($X_t$)** is accumulated linearly across layers.
   * The **Seasonal component ($X_s$)** is processed by attention and the CLM-subNet.
2. **Frequency-Domain Autocorrelation Mechanism:**
   * Uses Fast Fourier Transforms (FFT) to compute sequence similarities at periodic lag offsets:
     $$\tau_1, \dots, \tau_k = \text{argTopK}(\mathcal{R}_{Q,K}(\tau)), \quad k = \lfloor \log(L) \rfloor$$
   * Reduces computational complexity from quadratic $O(L^2)$ to efficient $O(L \log L)$.
3. **The CLM-subNet Innovation:**
   * Replaces the point-wise FFN in the Autoformer block with a **CNN-LSTM sub-network**.
   * **Two 1D-CNN Layers:** Act as localized high-frequency filters, scanning the seasonal sequence to detect abrupt peaks and drops.
   * **Bottleneck LSTM Layer:** Takes the CNN feature maps and compresses them to $64$ hidden units. Models the temporal progression of the extracted spikes.
   * **Efficiency Benefit:** Because the LSTM acts as a bottleneck ($d_{model} = 512 \to 64 \to 512$), CLM-Former actually has **fewer total parameters (33.96M)** than the official Autoformer (34.62M)!
4. **Decoder and Prediction Aggregation:**
   * The decoder refines the seasonal tokens using Cross-Autocorrelation with the encoder output, then applies the CLM-subNet.
   * The final prediction is generated by adding the projected seasonal output directly to the accumulated trend:
     $$\hat{X}_{pred} = w_s S_{de}^M + T_{de}^M$$

---

## 4. Key Lessons Learned and Practical Application for Review 2

### What the Literature Teaches Us:
1. **Decomposition is Non-Negotiable:** Raw power grid load must never be fed as an undivided sequence into deep networks. Trend (slow drift) and Seasonality (rapid oscillations) must be decomposed and handled by separate sub-modules.
2. **Hybrid Local-Global Architectures Win:** Pure attention captures global rhythm but misses local shocks; pure CNN/LSTM captures local spikes but struggles with long horizons. A hybrid that pairs attention/Mamba with CNN-LSTM achieves peak accuracy.
3. **Bottleneck Compressions Improve Efficiency:** Compressing intermediate representations before sequential recurrence (e.g., hidden size 64) prevents overfitting and lowers parameter count.

### Practical Replication in Our Codebase (`notebooks/02_Review2_Modeling_and_Forecasting.ipynb`):
To bridge the gap between academic theory and practical demonstration for Review 2, we replicated these architectural principles directly:
1. **Statistical Baselines:**
   * **ARIMA(1,1,1):** Baseline linear model without seasonal adaptation.
   * **SARIMA(1,1,1)x(1,0,1,7):** Incorporates the 7-day cyclical periodicity identified during our EDA.
2. **Machine Learning Baseline:**
   * **XGBoost:** Uses 7-day lookback lag features and calendar indicators.
3. **Deep Learning Innovation (CLM-subNet Architecture):**
   * Built in PyTorch using a **1D-CNN feature extractor** (`nn.Conv1d`, kernel size 3) connected to a **Bottleneck LSTM** (`nn.LSTM`, hidden size 64) and a linear projection head.
   * Evaluated across test horizons using strict metrics (**MAE**, **RMSE**, **MAPE**) to benchmark traditional statistical models against modern hybrid deep learning.
