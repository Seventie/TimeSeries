# Paper 2: CLM-Former — Convolutional-LSTM Transformer for Load Forecasting

**Full Title:** *CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model*  
**Authors:** S. Mozhgan Rahmatinia, Seyed-Majid Hosseini, Seyed-Amin Hosseini-Seno  
**Journal:** *Scientific Reports* (Nature Portfolio, 2026), Vol. 16, Article 4704  
**Primary Dataset Benchmark:** UCI Machine Learning Repository — ElectricityLoadDiagrams20112014 (321 residential clients, 26,305 hourly timesteps)  
**Generalizability Benchmarks:** ETTm2 (Electricity Transformer Temperature), Traffic (PEMS), Weather.

---

## 1. Executive Summary & Problem Formulation
Accurate multi-horizon electricity load forecasting is vital for smart microgrids to prevent blackouts, optimize demand response, and balance renewable energy distribution. 
Electricity consumption data possesses two opposing characteristics:
1. **Strong Long-Term Seasonality:** Deterministic 24-hour daily cycles and 7-day weekly cycles (workdays vs. weekend shutdowns).
2. **Sharp Local Volatility:** Unpredictable, high-frequency spikes and sudden drops caused by individual consumer habits, weather fluctuations, or equipment usage.

### The Critical Gap
* **Statistical Baselines (ARIMA/SARIMA):** Suffer from severe parameter explosions when handling high-frequency data and cannot model non-linear dynamic interactions.
* **Recurrent Architectures (LSTM/GRU/CNN-LSTM):** Suffer from sequential processing latency, gradient vanishing, and inability to maintain memory over very long horizons ($L > 100$).
* **Vanilla Transformers:** Use standard self-attention with quadratic complexity $O(L^2)$ and struggle to attend to localized fine-grained fluctuations.
* **Autoformer (Wu et al., 2021):** Solved the $O(L^2)$ bottleneck by replacing self-attention with **Frequency-Domain Autocorrelation** ($O(L \log L)$). However, Autoformer routes the seasonal component through a simple **point-wise Feed-Forward Network (FFN)**. Point-wise FFNs process each time step independently, completely destroying localized temporal dependencies and ignoring sharp, abrupt fluctuations.

**The CLM-Former Innovation:**  
The authors replace the point-wise FFN with a specialized **CLM-subNet (1D-CNN + Bottleneck LSTM)** to explicitly process the decomposed seasonal component, creating a true cross-domain synergy between frequency-domain periodicity and time-domain local volatility.

---

## 2. Detailed Architectural Blueprint of CLM-Former

CLM-Former employs an **Encoder-Decoder architecture** with integrated Series Decomposition:

### Block 1: Multi-Scale Additive Series Decomposition
Rather than treating the raw time series as a single entangled signal, CLM-Former decomposes the sequence at every encoder and decoder layer into two parts:
$$X_t = \text{AvgPooling}(\text{Padding}(X))$$
$$X_s = X - X_t$$
* **$X_t \in \mathbb{R}^{L \times d}$ (Trend-Cyclical Component):** Captures slow, long-term shifts in energy demand. Processed with lightweight moving averages.
* **$X_s \in \mathbb{R}^{L \times d}$ (Seasonal Component):** Captures high-frequency fluctuations and periodic oscillations. Processed by Autocorrelation and the CLM-subNet.

### Block 2: Frequency-Domain Autocorrelation Mechanism
Instead of computing scalar dot-product attention between all individual tokens ($O(L^2)$), Autocorrelation calculates cross-correlation across entire sub-series using the Wiener-Khinchin theorem and the **Fast Fourier Transform (FFT)** in $O(L \log L)$ time:
1. Compute Autocorrelation coefficients:
   $$\mathcal{R}_{Q,K}(\tau) = \lim_{L \to \infty} \frac{1}{L} \sum_{t=1}^L Q_t K_{t-\tau}$$
2. Select top-$k$ most significant lag periods:
   $$\tau_1, \dots, \tau_k = \text{argTopK}(\mathcal{R}_{Q,K}(\tau)), \quad k = \lfloor \log(L) \rfloor$$
3. Apply Softmax to lag similarities and roll values:
   $$\text{AutoCorrelation}(Q, K, V) = \sum_{i=1}^k \text{Roll}(V, \tau_i) \cdot \text{Softmax}(\mathcal{R}_{Q,K}(\tau_i))$$
This identifies repeating daily/weekly cycles regardless of phase shifts.

### Block 3: The CLM-subNet Architecture (The Hierarchical Bottleneck)
The core innovation of the paper resides in replacing the standard point-wise Feed-Forward Network (FFN) with the **CLM-subNet**:
1. **Two Stacked 1D-CNN Layers (Local Feature Detectors):**
   * Layer 1: Convolves over the seasonal sequence, projecting from dimension $d_{model} = 512$ to an intermediate feature dimension with Dropout ($p=0.05$).
   * Layer 2: Convolves back to $d_{model}$, extracting localized high-frequency peak-and-valley patterns.
2. **Bottleneck LSTM Layer (Temporal Sequence Modeling):**
   * The extracted CNN feature maps are fed into an LSTM with a compact hidden size of **$64$ units**.
   * Regulates information flow via input gate $i_t$, forget gate $f_t$, cell state $c_t$, and output gate $o_t$.
   * **The Bottleneck Effect:** By compressing features from $d_{model} = 512 \to 64$, the LSTM preserves long-term dependencies while *reducing* total model parameters from 34.62M (standard Autoformer) down to **33.96M (CLM-Former)**!
3. **Linear Projection:**
   * Projects the 64-dimensional LSTM representation back to $d_{model}$.

### Block 4: Encoder-Decoder Dataflow
* **Encoder (4 Layers):**
  $$S_{en}^{l,1}, \_ = \text{Decompose}(\text{AutoCorrelation}(X_{en}^{l-1}) + X_{en}^{l-1})$$
  $$S_{en}^{l,2}, \_ = \text{Decompose}(\text{CLM-subNet}(S_{en}^{l,1}) + S_{en}^{l,1})$$
* **Decoder (4 Layers):**
  Processes decomposed seasonal tokens $S_{de}$ and trend tokens $T_{de}$ initialized from historical averages.
  Incorporates **Cross-Autocorrelation Attention** between the decoder seasonal state and the encoder's seasonal output $S_{en}^N$.
* **Final Aggregation:**
  $$\hat{X}_{pred} = w_s S_{de}^M + T_{de}^M$$
  The seasonal prediction is projected and directly summed with the accumulated trend component.

---

## 3. Training & Hyperparameter Configuration
* **Input Lookback Length ($L$):** $128$ time steps.
* **Label Horizon:** $48$ steps.
* **Prediction Horizons ($T$):** $96$, $192$, and $336$ time steps.
* **Optimization:** Adam optimizer with MSE Loss, batch size $64$, early stopping with patience $3$.
* **Layer Depth:** $4$ Encoder layers, $4$ Decoder layers, $2$ LSTM layers in encoder CLM-subNet, $1$ LSTM layer in decoder CLM-subNet.

---

## 4. Key Experimental Results & Benchmarks
The paper evaluates CLM-Former on the **UCI Electricity dataset (321 clients)** across horizons $T \in \{96, 192, 336\}$:

| Model Category | Model | MSE (Horizon 96) | MSE (Horizon 192) | MSE (Horizon 336) | Average MSE | Average MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed** | **CLM-Former** | **0.196** | **0.203** | **0.229** | **0.209** | **0.323** |
| Transformer | Autoformer (2022) | 0.201 | 0.222 | 0.231 | 0.218 | 0.329 |
| Transformer | CrossFormer (2023) | 0.219 | 0.231 | 0.246 | 0.232 | 0.324 |
| Transformer | Informer (2021) | 0.274 | 0.296 | 0.300 | 0.290 | 0.382 |
| Transformer | Vanilla Transformer | 0.260 | 0.266 | 0.280 | 0.268 | 0.366 |
| Deep Learning | TiDE (2023) | 0.237 | 0.236 | 0.249 | 0.240 | 0.334 |
| Deep Learning | SCINet (2022) | 0.247 | 0.257 | 0.269 | 0.257 | 0.356 |
| Deep Learning | Standard CNN-LSTM | 0.353 | 0.409 | 0.412 | 0.391 | 0.433 |
| Deep Learning | Standard LSTM | 0.377 | 0.443 | 0.438 | 0.419 | 0.460 |

### Critical Takeaways:
1. **46.5% error reduction over standard CNN-LSTM:** Proves that simply stacking CNN and LSTM sequentially fails, but embedding them *inside a decomposition block* achieves state-of-the-art results.
2. **Superiority over Autoformer:** Achieves a 5.16% reduction in MAE at horizon 336 while running with **fewer parameters** and equal inference speed.
