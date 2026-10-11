# Paper 1: MKJR — Mamba with Kolmogorov-Arnold Network and Jacobian Regularization

**Full Title:** *A unified neural framework for long-term time series forecasting and granger-style causal analysis*  
**Authors:** Yanan Zhang, Bo Feng, Zan Chen, Shen Zhang  
**Journal:** *Engineering Applications of Artificial Intelligence* (Elsevier, 2026), Vol. 176, Article 114802  
**Dataset Benchmarks:** UCI Electricity (321 clients), Traffic (862 channels), Weather (21 variables), ETT (ETTh1, ETTh2, ETTm1, ETTm2), and Industrial Causal Benchmarks (Tennessee Eastman, Ultra-Processed Food, Debutanizer).

---

## 1. Executive Summary & Problem Formulation
In real-world multivariate time series (like electricity grids, industrial chemical plants, and traffic sensor networks), practitioners face two intertwined challenges:
1. **Long-Term Time Series Forecasting (LTSF):** Accurately predicting future horizons ($T \in \{96, 192, 336, 720\}$ steps) given historical lookbacks ($L = 96$).
2. **Causal & Dependence Analysis:** Identifying *which* variables and *which historical lags* actually cause changes in other variables (Granger-style causal structure discovery).

### The Critical Gap
Traditionally, researchers treated these two problems separately:
* Forecasting models (Transformers, LSTMs, Linear models) maximize prediction accuracy but operate as uninterpretable "black boxes" that cannot output causal graphs.
* Causal discovery models (cMLP, cLSTM, VAR, TCDF) recover causal relationships but scale poorly to long-sequence forecasting and high dimensions.

**The MKJR Innovation:**  
The authors propose **MKJR**, a unified architecture where a **single trained multivariate predictor** simultaneously provides state-of-the-art long-term forecasts and extracts lag-resolved Granger-style causal dependency graphs.

---

## 2. Detailed Architectural Blueprint of MKJR

The MKJR architecture consists of five core mathematical blocks:

### Block 1: Reversible Instance Normalization (RevIN)
To handle severe non-stationarity and distribution shifts across time windows:
$$X^{(0)} = \text{Norm}(X) \in \mathbb{R}^{B \times M \times L}$$
RevIN standardizes each instance by its local mean and standard deviation, and later reverses the scaling at the final output.

### Block 2: Kolmogorov-Arnold Network (KAN) Encoder
Traditional Multi-Layer Perceptrons (MLPs) use fixed activation functions ($\text{ReLU}, \text{GeLU}$) on nodes with learnable scalar weights on edges. In contrast, **KANs** place **learnable univariate spline functions on the edges**:
$$\phi(x) = w_{base} b(x) + \sum_i c_i B_i(x)$$
Where $B_i(x)$ are $B$-spline basis functions (order $k=3$, grid interval $G=5$).
* **Two-Stage Encoding:**
  $$Z^{(1)} = \text{KE}_1(X^{(0)}), \quad Z^{(2)} = \text{KE}_2(\text{Drop}(Z^{(1)}))$$
  with latent dimensions $(N_1, N_2) = (256, 128)$.
* **Encoder Structural Prior ($\pi_{i,\alpha}$):**  
  Because KAN edges directly connect input lag $\alpha$ of variable $i$ to hidden units, the model computes the retention strength of each lag:
  $$\pi_{i,\alpha} = \frac{\left( \sum_{h=1}^{N_1} [(w^{base}_{h,i,\alpha})^2 + \|c^{spline}_{h,i,\alpha,:}\|_2^2] \right)^{1/2}}{\sum_{\beta=1}^L \left( \sum_{h=1}^{N_1} [(w^{base}_{h,i,\beta})^2 + \|c^{spline}_{h,i,\beta,:}\|_2^2] \right)^{1/2} + \varepsilon}$$
  This serves as an encoder-side prior of which past lags matter.

### Block 3: Asymmetric Bidirectional Dual-Stream Mamba Backbone
Rather than quadratic-complexity Transformers ($O(L^2)$), MKJR uses **Mamba State-Space Models (SSMs)** ($O(L)$ linear complexity).
MKJR processes representations across two asymmetric streams:
1. **Detail Stream (Operates on $Z^{(1)}$):** Modulated by local temporal signatures extracted with a 1D-CNN ($\Phi_{spec}$, kernel size 3):
   $$\tilde{Z}_{loc} = Z^{(1)} \odot G_{loc}$$
   *No residual shortcut is added*, forcing the detail stream to focus strictly on localized high-frequency features.
2. **Global Stream (Operates on $Z^{(2)}$):** Modulated by global variable-wise pooling:
   $$\tilde{Z}_{glo} = Z^{(2)} \odot G_{glo}$$
   *Retains an identity residual shortcut* $+ Z^{(2)}$ to stabilize long-range cross-variable dependencies.
3. **Bidirectional Scanning:** Each stream passes through both a **forward Mamba block** and a **reversed Mamba block**:
   $$H^{(1)} = \text{MB}^{loc}_1(\tilde{Z}_{loc}) + \text{Rev}(\text{MB}^{loc}_2(\text{Rev}(\tilde{Z}_{loc})))$$
4. **Modified Mamba Block:** Removes standard convolutions and replaces them with a KAN Projection ($K_P$), Bag-of-Activations ($\Psi_{BoA}$), and Dynamic BoA Gate ($G_{DB}$).

### Block 4: KAN Decoder
A two-stage KAN decoder ($\text{KD}_1, \text{KD}_2$) decodes the fused representations into future horizons:
$$\hat{Y} = \text{KD}_2(\text{Concat}(H^{(1)}, \text{KD}_1(H^{(2)}) + Z^{(1)}))$$
The decoder weights provide a target-specific variable-level readout preference score ($s^{dec}_{j,i}$).

### Block 5: Matrix-Free Random-Projection Jacobian Regularization (Training Time)
To prevent overfitting and force the model to learn smooth, genuine causal dependencies rather than spurious noise, the authors penalize the model's input-output sensitivity.
Computing the full Jacobian matrix $J = \frac{\partial \hat{Y}}{\partial X}$ requires $O(M^2 \cdot L \cdot T)$ operations, which causes Out-Of-Memory crashes.
**The Matrix-Free Solution:** They project predictions onto random probe vectors (Hutchinson estimator):
* **Gaussian Sensitivity Penalty ($R_{sens}$):** Using probe $v \sim \mathcal{N}(0, I)$:
  $$R_{sens} = \frac{1}{B \cdot n_{proj}} \sum_{b=1}^B \sum_{\mu=1}^{n_{proj}} \|\nabla_{X_b} (v_b^{(\mu)\top} \hat{Y}_b)\|_2^2$$
* **Rademacher Structural Sparsity Penalty ($R_{str}$):** Using probe $p \sim \text{Rademacher}(\pm 1)$ combined with KAN lag sparsity.
* **Overall Loss:**
  $$\mathcal{L} = \text{MSE} + \lambda_{sens} R_{sens} + \lambda_{str} R_{str} \quad (\lambda_{sens} = 0.10, \lambda_{str} = 0.10)$$

---

## 3. Evaluation-Time Granger Causal Graph Recovery
Once trained, the model extracts the causal graph without retraining:
1. **Lag-Resolved Causal Score:**
   $$Y^{causal}_{j \leftarrow i, \alpha} = \pi_{i,\alpha} \cdot \frac{1}{T} \sum_{u=1}^T \left| \frac{\partial \hat{y}_{t+u,j}}{\partial x_{t+1-\alpha,i}} \right|$$
   Combines KAN encoder structural prior $\pi_{i,\alpha}$ with Jacobian sensitivity.
2. **Pairwise Edge Score:**
   $$C_{j \leftarrow i} = s^{dec}_{j,i} \cdot \frac{1}{\sqrt{L}} \|Y^{causal}_{j \leftarrow i, :}\|_2$$
3. **Surrogate Permutation Significance Test:**  
   Generates circular-shift surrogate trajectories ($B_{surr} = 100$) and applies **Benjamini-Hochberg False Discovery Rate (FDR)** control ($\alpha = 0.05$) to output a binary Directed Acyclic / Adjacency Graph $A \in \{0, 1\}^{M \times M}$.

---

## 4. Key Experimental Results & Benchmarks
* On the **UCI Electricity dataset** (321 variables): MKJR achieves average MSE of **0.162** and MAE of **0.257**, outperforming iTransformer (MSE 0.178), TimeMachine (0.170), S-Mamba (0.170), and DLinear (0.221).
* In causal discovery benchmarks (VAR, fMRI, DREAM-3, Tennessee Eastman), MKJR achieves near-perfect AUROC ($1.000$ on VAR, $0.931$ on fMRI, $0.926$ on Medical), proving that regularizing gradient sensitivity directly reveals true physical relationships.
