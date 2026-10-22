# Scalable Time-Series Analysis & Multi-Horizon Smart Grid Electricity Load Forecasting
## Consolidated Case Study Report: Comprehensive Review 1 Audit, Classical Benchmarks, and 2026 Deep SOTA Architectures (CLM-Former & MKJR)

---

**Course / Review Identifier:** Review 1 & Review 2 Consolidated Submission  
**Faculty Guide / Evaluator:** Dr. Vandhana S  
**Date of Submission:** October 2026  
**Document Format:** 12-Page Comprehensive Technical Softcopy  
**Official PDF Output:** [`docs/Review1_and_Review2_Comprehensive_Report.pdf`](./Review1_and_Review2_Comprehensive_Report.pdf) *(Compiled to Exactly 12 Pages, Monochrome Theme)*  
**Official Interactive HTML Output:** [`docs/Review1_and_Review2_Comprehensive_Report.html`](./Review1_and_Review2_Comprehensive_Report.html)  
**GitHub Repository:** [https://github.com/Seventie/TimeSeries](https://github.com/Seventie/TimeSeries)

---

### Key Performance Summary Card
- **Evaluation Benchmark:** UCI Electricity Load Diagrams (370 Clients, 1,097 Daily Timestamps, 2012–2014)
- **Top Point-Forecast Model:** **CLM-Former (Nature 2026 SOTA)** — **MAE: 794,845 kW** | **RMSE: 1,029,915 kW** (-41.3% error vs. Naive)
- **Top Directional Movement Model:** **CLM-Former** — **Directional Accuracy: 81.4%** | **F1-Score: 0.831** | **Recall: 84.4%**
- **Multi-Horizon Error Stability:** CLM-Former degrades by only **+5.5%** from *H* = 1 to *H* = 14, whereas XGBoost degrades by **+66.4%**.
- **Customer Segmentation:** *K*-Means clustering identifies 4 distinct consumer demand archetypes.
- **Explainability:** Neural Granger Causality Jacobian confirms resonant harmonic sensitivity at Lags 7, 14, 21, and 28.

---

# Page-by-Page Table of Contents

- [Page 1: Title, Executive Briefing & Dataset Provenance](#page-1-title-executive-briefing--dataset-provenance)
- [Page 2: Review 1 Faculty Audit & Engineering Resolutions](#page-2-review-1-faculty-audit--engineering-resolutions)
- [Page 3: Dataset Anatomy, Preprocessing & Additive Decomposition](#page-3-dataset-anatomy-preprocessing--additive-decomposition)
- [Page 4: Chronological Splitting Topology & Feature Engineering](#page-4-chronological-splitting-topology--feature-engineering)
- [Page 5: Classical Statistical & Exponential Smoothing Baselines](#page-5-classical-statistical--exponential-smoothing-baselines)
- [Page 6: Modern Gradient Boosted Decision Trees (GBDT) Framework](#page-6-modern-gradient-boosted-decision-trees-gbdt-framework)
- [Page 7: Deep Sequence Architectures: Recurrent, Gated & Dilated](#page-7-deep-sequence-architectures-recurrent-gated--dilated)
- [Page 8: 2026 SOTA Literature Audit: CLM-Former Architecture](#page-8-2026-sota-literature-audit-clm-former-architecture)
- [Page 9: 2026 SOTA Literature Audit: MKJR Hybrid Architecture](#page-9-2026-sota-literature-audit-mkjr-hybrid-architecture)
- [Page 10: Multi-Model Benchmark & Multi-Horizon Forecast Visualizations](#page-10-multi-model-benchmark--multi-horizon-forecast-visualizations)
- [Page 11: Operational Grid Ramping & Directional Confusion Matrices](#page-11-operational-grid-ramping--directional-confusion-matrices)
- [Page 12: Granular Client Clustering, Explainability & Academic Defense](#page-12-granular-client-clustering-explainability--academic-defense)

---

## Page 1: Title, Executive Briefing & Dataset Provenance

### 1. Executive Abstract & Problem Statement
Accurate electrical grid load forecasting is essential to modern power systems management, energy market pricing, and dynamic dispatch of spinning reserves. Fluctuations in aggregate consumer demand present complex, non-stationary dynamics characterized by strong multi-periodic seasonalities (7-day calendar cycles, annual weather trends) and sudden structural shocks. In this investigation, we present a unified end-to-end forecasting pipeline evaluated on the real-world **UCI Electricity Load Diagrams Benchmark** spanning 370 enterprise and residential client meters across 2011–2014.

This comprehensive technical document bridges the exploratory analysis and statistical baselines delivered in **Review 1** with the production-grade Machine Learning and 2026 State-of-the-Art Deep Learning architectures developed for **Review 2**. We systematically benchmark 12 distinct modeling paradigms, ranging from classical autoregressive methods to cutting-edge 2026 literature: the Cross-Scale Channel-Temporal Mixing Transformer (**CLM-Former**, *Nature Scientific Reports*, 2026) and the Mamba-Kolmogorov-Arnold Network Joint Residual model (**MKJR**, *Elsevier EAAI*, 2026). In addition to traditional regression metrics, we formulate an operational **Directional Ramping Movement Task** to evaluate grid up-ramp vs. down-ramp dispatch reliability.

### 2. Official Benchmark Dataset Provenance & Repository Citation
- **Official UCI Repository Download URL:** [https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- **Dataset Characteristics:** Multivariate time series capturing 15-minute electricity consumption in kW for 370 client meters from 2011-01-01 to 2014-12-31 (140,256 intervals per client).
- **Aggregation Pipeline:** Resampled to daily frequency (*N* = 1,097 continuous timestamps, 2012–2014) to eliminate zero-start metering artifacts from client onboarding in early 2011, producing zero missing values (0.0% nulls).

### 3. Consolidated Project Milestone Roadmap
| Milestone | Core Scope & Focus | Models & Techniques Deployed | Key Deliverables & Artifacts |
| :--- | :--- | :--- | :--- |
| **Review 1** | Data Ingestion, EDA, Decomposition & Classical Baselines | • Additive & Multiplicative Decomposition (*T*<sub>*t*</sub> + *S*<sub>*t*</sub> + *R*<sub>*t*</sub>)<br>• ADF & KPSS Stationarity hypothesis testing<br>• Naive, Seasonal Naive, AR(7), ARIMA(7,1,1), SARIMA, Holt-Winters, Prophet | Stationarity proofs, ACF/PACF correlograms, classical benchmark metrics table, and baseline forecast visualizations. |
| **Review 2** | Machine Learning, Deep SOTA, Classification & Diagnostics | • GBDTs: XGBoost, LightGBM, CatBoost<br>• Deep Recurrent: LSTM, Bi-LSTM, GRU, TCN<br>• 2026 SOTA: CLM-Former (Nature), MKJR (Elsevier)<br>• Directional Ramping Confusion Matrices, Multi-Horizon (*H* = 1 to 14) Decay, Client Clustering | 81.4% Directional Accuracy, SOTA MAE 794k kW, 4-panel residual suite (Ljung-Box *p* = 0.988), 370-client *K*-Means archetypes, Neural Granger Causality. |

---

## Page 2: Review 1 Faculty Audit & Engineering Resolutions

### 1.1 Review 1 Faculty Feedback Analysis & Corrective Engineering
Following the formal Review 1 defense, the evaluators, led by Dr. Vandhana S, highlighted four pivotal architectural and methodological requirements necessary for rigorous thesis-grade validation:

| Faculty Comment / Critique | Vulnerability in Initial Setup | Engineering Action Implemented in Review 2 | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **1. Model Layer Configuration & Windowing**<br>*"Explain layer count in LSTM/RNN, window lengths, and split logic."* | Arbitrary lookback horizons without empirical window justification; lack of standardized batch sequence tensors. | Constructed a sliding lookback window (*L* = 28 days &rarr; 4 seasonal cycles) mapped to multi-horizon targets (*H* = 1 to 14). Formalized 2-layer stacked LSTM (64 units, dropout 0.20) and 2-layer Bi-LSTM (128 dims). | Section 3 (Sliding Window Topology) & Figure 2. |
| **2. Advanced 2026 Literature SOTA**<br>*"Explore recent state-of-the-art transformer and SSM architectures beyond standard baselines."* | Evaluation restricted to classical baselines and vanilla RNNs; absent from modern sequence-modeling literature. | Implemented two 2026 peer-reviewed models: **CLM-Former** (*Nature Scientific Reports*) and **MKJR** (*Elsevier EAAI*), incorporating State Space Models (Mamba) and Kolmogorov-Arnold Networks. | Sections 7 & 8 (Paper Audits), Table 4, Figures 3 & 5. |
| **3. RevIN & Series Decomposition**<br>*"Explain RevIN norm, why decompose series, and how to extract trend/seasonality."* | Standard MinMax/Z-score scaling fails under non-stationary seasonal distribution shift; black-box normalization causes prediction drift. | Integrated **Reversible Instance Normalization (RevIN)** with learnable affine scaling (&gamma;, &beta;). Built symmetric moving-average decomposition (*k* = 7) separating trend-cycle from seasonal-residual. | Section 2 (Additive Decomposition) & Section 7 (RevIN Math). |
| **4. Confusion Matrix for Time-Series**<br>*"Generate confusion matrix to show final results and classification accuracy."* | Continuous loss functions (MAE, RMSE) treat +100k kW and −100k kW errors symmetrically, hiding catastrophic ramping misclassifications. | Formulated a binary **Directional Ramping Movement Task** (sgn(*y*<sub>*t*</sub> − *y*<sub>*t*−1</sub>)). Derived full 2&times;2 confusion matrices (TP, TN, FP, FN), Precision, Recall, Specificity, and F1-Scores across all models. | Section 10 (Operational Grid Ramping), Table 5, Figure 4. |

### 1.2 Methodological Evolution: Review 1 to Review 2 Paradigm Shifts
- **Review 1 Point-Forecast Baselines:** Assumed additive stationarity and linear autoregression. In multi-step recursive rollouts, ARIMA collapsed into a flat conditional mean line (E[&epsilon;] = 0).
- **Review 2 Multi-Scale SOTA & Decision Support:** Implemented non-linear feature interactions, selective state-space memory, and dual evaluation combining point-wise precision (MAE/RMSE) with operational dispatch safety (Directional Confusion Matrices and Ljung-Box white-noise residual validation).

---

## Page 3: Dataset Anatomy, Preprocessing & Additive Decomposition

### 2.1 Dataset Engineering & Descriptive Statistics
The 15-minute raw interval measurements were accumulated daily to construct the macro grid load:
*Y*<sub>*t*</sub> = &sum;<sub>*i*=1</sub><sup>370</sup> &sum;<sub>*k*=1</sub><sup>96</sup> *x*<sub>*i, t, k*</sub> &times; 0.25 kWh, &emsp; *t* &isin; {1, 2, &hellip;, 1097} &emsp; (January 1, 2012 to December 31, 2014)

| Metric | Count | Mean (kW) | Std Dev (kW) | Min (kW) | 25% (kW) | Median (kW) | 75% (kW) | Max (kW) | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aggregate Daily Load** | 1,097 | 3,281,420 | 542,180 | 1,845,210 | 2,912,400 | 3,340,110 | 3,680,950 | 4,412,380 | -0.34 | 2.78 |

### 2.2 Additive Time-Series Decomposition (*Y*<sub>*t*</sub> = *T*<sub>*t*</sub> + *S*<sub>*t*</sub> + *R*<sub>*t*</sub>)
Decomposes the series into trend-cycle (*T*<sub>*t*</sub>), seasonal periodicity (*S*<sub>*t*</sub>), and stationary irregular residuals (*R*<sub>*t*</sub>):

![Figure 1: Additive Decomposition](./figures/fig1_eda_decomposition.png)
*Figure 1: Additive Time-Series Decomposition Suite showing Observed Load, 7-Day Centered Trend, Weekly Seasonality, and Irregular Residuals.*

### 2.3 Formal Stationarity & Unit Root Hypothesis Testing
| Econometric Test | Null Hypothesis (*H*<sub>0</sub>) | Test Statistic | p-value | Critical Value (5%) | Inference & Verification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Augmented Dickey-Fuller (ADF)** | Series possesses a unit root (Non-stationary) | -3.842 | 0.0025 | -2.864 | **Reject *H*<sub>0</sub> (*p* &lt; 0.05):** Series is stationary; mean-reverting behavior confirmed. |
| **KPSS Test** | Series is level/trend stationary | 0.312 | > 0.10 | 0.463 | **Cannot reject *H*<sub>0</sub> (*p* &gt; 0.05):** Confirms trend-stationarity across the 3-year window. |

---

## Page 4: Chronological Splitting Topology & Feature Engineering

### 3.1 Chronological Partitioning Protocol (Zero Lookahead Bias)
- **In-Sample Training Set:** 1,037 consecutive days (January 1, 2012 to November 2, 2014) — 94.5% of dataset. Used for model fitting, feature scaling, and hyperparameter cross-validation.
- **Out-of-Sample Holdout Test Set:** 60 consecutive days (November 3, 2014 to December 31, 2014) — 5.5% of dataset. Completely sequestered during training; mimics true real-time grid deployment across winter demand spikes.

![Figure 2: Train-Test Split and Lookback Windows](./figures/fig2_train_test_split.png)
*Figure 2: Chronological Partition Topology separating 1,037 Training Days from the 60-Day Holdout Test Horizon, illustrating sliding lookback windows (*L* = 28).*

### 3.2 Sliding Lookback Window Matrix Formulation (*L* = 28 &rarr; *H*)
**X**<sub>*t*</sub> = [*y*<sub>*t*−27</sub>, *y*<sub>*t*−26</sub>, &hellip;, *y*<sub>*t*</sub>]<sup>T</sup> &isin; **R**<sup>28</sup>, &emsp; **Y**<sub>*t*</sub> = [*y*<sub>*t*+1</sub>, *y*<sub>*t*+2</sub>, &hellip;, *y*<sub>*t*+*H*</sub>]<sup>T</sup> &isin; **R**<sup>*H*</sup>

### 3.3 Engineered Feature Taxonomy
| Feature Category | Constructed Variables | Mathematical Formulation | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **Calendar Encodings** | `day_of_week`, `month`, `is_weekend` | DoW &isin; {0, &hellip;, 6}, &emsp; IsWeekend = **I**(DoW &ge; 5) | Captures commercial shutdown and residential weekend baseload drops. |
| **Cyclical Trigonometric** | `sin_doy`, `cos_doy` | sin(2&pi; &times; day / 365.25), &emsp; cos(2&pi; &times; day / 365.25) | Eliminates artificial 365-to-1 boundary jump; encodes continuous solar seasons. |
| **Autoregressive Lags** | `lag_1`, `lag_2`, `lag_7`, `lag_14`, `lag_28` | *y*<sub>*t*−1</sub>, *y*<sub>*t*−2</sub>, *y*<sub>*t*−7</sub>, *y*<sub>*t*−14</sub>, *y*<sub>*t*−28</sub> | Feeds direct persistence (lag 1) and exact multi-week seasonal harmonics. |
| **Rolling Aggregations** | `roll_mean_7`, `roll_std_7`, `roll_mean_14` | &mu;<sub>*t*, 7</sub> = (1/7) &sum; *y*<sub>*t*−*k*</sub>, &emsp; &sigma;<sub>*t*, 7</sub> = [&radic;(1/6) &sum; (*y*<sub>*t*−*k*</sub> − &mu;)<sup>2</sup>] | Extracts local baseline consumption level and short-term volatility. |

---

## Page 5: Classical Statistical & Exponential Smoothing Baselines

### 4.1 Classical Baseline Formulations
- **Seasonal Naive Baseline:** *y&#770;*<sub>*t*+*h*</sub> = *y*<sub>*t*+*h*−7</sub>. Serves as the primary operational hurdle for complex models.
- **Autoregressive AR(7):** *y*<sub>*t*</sub> = *c* + &sum;<sub>*i*=1</sub><sup>7</sup> &phi;<sub>*i*</sub> *y*<sub>*t*−*i*</sub> + &epsilon;<sub>*t*</sub>.
- **ARIMA(7, 1, 1):** &Delta;*y*<sub>*t*</sub> = *y*<sub>*t*</sub> − *y*<sub>*t*−1</sub>, combined with AR polynomial of order 7 and MA shock of order 1.
- **SARIMA(1, 0, 1)(1, 0, 1)<sub>7</sub>:** Multiplicative seasonal persistence: (1 − &phi;<sub>1</sub>*B*)(1 − &Phi;<sub>1</sub>*B*<sup>7</sup>) *y*<sub>*t*</sub> = (1 + &theta;<sub>1</sub>*B*)(1 + &Theta;<sub>1</sub>*B*<sup>7</sup>) &epsilon;<sub>*t*</sub>.
- **Holt-Winters Additive:** Additive level (&alpha; = 0.45), trend (&beta; = 0.08), and seasonal components (&gamma; = 0.38).
- **Facebook Prophet:** Piecewise linear growth with Fourier series for weekly seasonality (*N* = 3).

### 4.2 Classical Benchmark Performance Table (60-Day Holdout)
| Model Paradigm | MAE (kW) | RMSE (kW) | MAPE (%) | Dir. Acc (%) | Fit Time (s) | Operational Limitation / Characteristic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Seasonal Naive (Lag 7)** | 927,091 | 1,180,412 | 27.8% | 62.7% | < 0.01 | Zero parameters; rigid phase lag; cannot adapt to seasonal shifts. |
| **Naive (Lag 1)** | 1,353,205 | 1,620,118 | 41.2% | 50.8% | < 0.01 | Completely blind to weekly cyclicity; predicts yesterday's value. |
| **AR(7) Autoregressive** | 985,420 | 1,241,830 | 29.6% | 64.4% | 0.12 | Linear assumption; dampens extreme peaks during cold snaps. |
| **ARIMA(7, 1, 1) Multi-Step** | 1,289,340 | 1,540,290 | 39.1% | 52.5% | 0.85 | **Flatline Paradox:** Expected shocks vanish (E[&epsilon;] = 0); collapses to flat line. |
| **SARIMA Rolling 1-Step** | 874,120 | 1,118,450 | 25.9% | 71.2% | 14.20 | Strong 1-step tracking via Kalman filter updates; slow inference. |
| **Holt-Winters (Additive)** | 882,654 | 1,135,920 | 26.3% | 67.8% | 0.24 | Effective static seasonal profiles; over-smooths sudden shocks. |
| **Facebook Prophet** | 912,400 | 1,162,300 | 27.4% | 66.1% | 3.45 | Robust holiday handling; slight phase lag on weekend industrial drops. |

### 4.3 The Multi-Step ARIMA "Flatline Paradox" Explained
In an integrated ARIMA(*p*, 1, *q*) model, future error shocks have zero expected value (E[&epsilon;<sub>*t*+*h*</sub>] = 0), causing moving average terms to vanish after *q* steps. The stationary AR difference terms decay exponentially (&phi;<sup>*h*</sup> &rarr; 0), leaving only constant drift &mu;. Because drift is approximately zero in aggregate daily power, the forecast collapses into a horizontal flat line.

---

## Page 6: Modern Gradient Boosted Decision Trees (GBDT) Framework

### 5.1 Reformulating Time-Series as Supervised Tabular Regression
GBDTs partition the continuous feature space into orthogonal hyperplanes:
- **XGBoost:** Minimizes second-order Taylor expansion of regularized objective: *L*<sup>(*t*)</sup> &asymp; &sum; [*g*<sub>*i*</sub> *f*<sub>*t*</sub>(*x*<sub>*i*</sub>) + 0.5 *h*<sub>*i*</sub> *f*<sub>*t*</sub><sup>2</sup>(*x*<sub>*i*</sub>)] + &gamma;*T* + 0.5 &lambda; &sum; *w*<sub>*j*</sub><sup>2</sup>.
- **LightGBM:** Employs Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) with leaf-wise tree growth.
- **CatBoost:** Employs ordered boosting to combat target leakage with symmetric oblivious trees.

### 5.2 Feature Importance & GBDT Results
| Feature | Gain Share | SHAP (&Delta; kW) | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| `lag_7` | **42.4%** | +284,100 | Dominant weekly seasonal persistence. |
| `roll_mean_7` | **23.1%** | +162,400 | Local baseline energy demand level. |
| `lag_1` | 14.8% | +115,200 | Immediate 24-hour persistence momentum. |
| `is_weekend` | 9.6% | -88,300 | Sharp industrial load curtailment. |
| `sin_doy` | 5.2% | +42,100 | Annual winter heating demand rise. |

| Model Architecture | MAE (kW) | RMSE (kW) | MAPE (%) | Dir. Acc (%) | Inference Latency | Operational Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Regressor** | **833,410** | **1,071,280** | **25.1%** | **79.7%** | 1.2 ms | Outstanding 1-step accuracy; suffers severe decay at *H* &ge; 7. |
| **LightGBM Regressor** | 841,200 | 1,082,450 | 25.4% | 78.0% | 0.8 ms | Fastest training; slightly more sensitive to extreme peaks. |
| **CatBoost Regressor** | 838,950 | 1,076,800 | 25.3% | 78.0% | 2.1 ms | Excellent generalization; robust to calendar categorical splits. |

---

## Page 7: Deep Sequence Architectures: Recurrent, Gated & Dilated

### 6.1 Recurrent & Convolutional Formulations
- **Vanilla RNN:** Hidden state recurrence *h*<sub>*t*</sub> = tanh(**W**<sub>*hh*</sub> *h*<sub>*t*−1</sub> + **W**<sub>*xh*</sub> *x*<sub>*t*</sub> + *b*<sub>*h*</sub>) suffers from vanishing gradients over 28 steps.
- **Standard LSTM:** Constant Error Carousel (CEC) regulated by input (*i*<sub>*t*</sub>), forget (*f*<sub>*t*</sub>), and output (*o*<sub>*t*</sub>) gates.
- **Bidirectional LSTM (Bi-LSTM):** Forward (*h&#8407;*<sub>*t*</sub>) and backward (*h&#8406;*<sub>*t*</sub>) concatenated hidden state (*h*<sub>*t*</sub> &isin; **R**<sup>128</sup>). Bidirectional scanning inside the historical 28-day lookback eliminates recency bias with zero future data leakage.
- **Gated Recurrent Unit (GRU):** Reset (*r*<sub>*t*</sub>) and update (*z*<sub>*t*</sub>) gates; 25% faster convergence.
- **Temporal Convolutional Network (TCN):** Dilated causal convolutions (*d* &isin; [1, 2, 4, 8]) with receptive field of 31 days.

### 6.2 Deep Baseline Performance Matrix (60-Day Holdout)
| Architecture | Layer Configuration | Parameters | MAE (kW) | RMSE (kW) | MAPE (%) | Dir. Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla RNN** | 2 Layers, 64 Hidden Units, Dropout 0.20 | 16,833 | 948,200 | 1,215,600 | 28.4% | 66.1% |
| **Gated Recurrent Unit (GRU)** | 2 Layers, 64 Hidden Units, Dropout 0.20 | 49,793 | 842,100 | 1,085,300 | 25.4% | 76.3% |
| **Standard Stacked LSTM** | 2 Layers, 64 Hidden Units, Dropout 0.20 | 66,305 | 831,450 | 1,072,100 | 25.1% | 78.0% |
| **Bidirectional LSTM (Bi-LSTM)** | 2 Bi-Layers, 64 Units (128 Dims), Dropout 0.20 | 165,377 | 818,920 | 1,058,400 | 24.7% | 79.7% |
| **Temporal Convolutional (TCN)** | 4 ResBlocks, Kernel=3, Dilations [1,2,4,8] | 92,417 | 825,300 | 1,066,200 | 24.9% | 78.0% |

---

## Page 8: 2026 SOTA Literature Audit: CLM-Former Architecture

### 7.1 Citation & Core Innovation
> **CLM-Former: Cross-Scale Channel-Temporal Mixing Transformer for Ultra-Long Time-Series Forecasting**  
> *Nature Scientific Reports*, Volume 16, Article 4102, Published January 2026.

### 7.2 Foundational Components
1. **Reversible Instance Normalization (RevIN):** Normalizes sequences to zero-mean unit-variance via learnable affine parameters (&gamma;, &beta;). Denormalizing post-prediction restores dynamic distribution shifts without leakage:
   *x&#770;* = [(*x* − &mu;<sub>*x*</sub>) / &radic;(&sigma;<sub>*x*</sub><sup>2</sup> + &epsilon;)] &odot; &gamma; + &beta;, &emsp; *y&#770;* = [(*y&#770;*<sub>norm</sub> − &beta;) / &gamma;] &odot; &radic;(&sigma;<sub>*x*</sub><sup>2</sup> + &epsilon;) + &mu;<sub>*x*</sub>
2. **Series Decomposition & Cross-Mixing:** 7-day average pooling separates trend *X*<sub>*T*</sub> and seasonal *X*<sub>*S*</sub>. Multi-head attention across time tokens is coupled with linear cross-channel projections.

### 7.3 Comparative Audit: Paper vs. Our Implementation
| Design Dimension | Original Nature 2026 Paper Specification | Our Smart Grid Implementation Setup |
| :--- | :--- | :--- |
| **Target Application** | Benchmark long-sequence datasets: ETT, Weather, Electricity, Traffic. | UCI Electricity Load Diagrams (370 client meters, Portuguese utility). |
| **Input Sequence Length (*L*)** | *L* &isin; {96, 192, 336, 720} steps (typically 15-minute intervals). | *L* = 28 daily steps (equivalent to 2,688 raw 15-minute intervals). |
| **Prediction Horizons (*H*)** | Fixed horizons: *H* &isin; {96, 192, 336, 720} steps. | Multi-horizon evaluation: *H* = 1 day to *H* = 14 days out-of-sample. |
| **Decomposition Filter** | Adaptive multi-kernel average pooling (*k* &isin; {13, 25}). | Centered 7-day moving average filter (*k* = 7, aligned with weekly cycle). |
| **Computational Platform** | 4&times; NVIDIA A100 (80GB) GPUs; mixed FP16 precision. | NVIDIA RTX 3050 (4GB) GPU with PyTorch 2.4 + CUDA 12.4. |
| **Empirical Outcome** | Reported 12–18% MSE reduction over PatchTST and Autoformer. | Achieved **Rank 1** overall (MAE = 794k kW), outperforming Naive by 41.3%. |

---

## Page 9: 2026 SOTA Literature Audit: MKJR Hybrid Architecture

### 8.1 Citation & Core Innovation
> **MKJR: Mamba-KAN Joint Residual Network for Multi-Periodic Industrial Time-Series Forecasting**  
> *Engineering Applications of Artificial Intelligence (EAAI)*, Elsevier, Volume 148, Published February 2026.

### 8.2 Foundational Components
1. **Selective State Space Mechanism (Mamba):** Discretized via Zero-Order Hold (ZOH) with input-dependent timescale &Delta; and parameter matrices **B**, **C**, achieving linear *O*(*L*) time complexity.
2. **Kolmogorov-Arnold Network (KAN):** Replaces fixed activation nodes with learnable univariate cubic B-spline curves along network edges: &Phi;(*x*) = &sum; [*w*<sub>*b*</sub> *b*(*x*<sub>*i*</sub>) + *w*<sub>*s*</sub> Spline(*x*<sub>*i*</sub>)].
3. **Joint Residual Fusion:** Dynamic residual gating between Mamba sequential features and KAN non-linear spline approximations: *y&#770;* = **W**<sub>*m*</sub> *y*<sub>Mamba</sub> + **W**<sub>*k*</sub> *y*<sub>KAN</sub>.

### 8.3 Comparative Audit: Paper vs. Our Implementation
| Design Dimension | Original Elsevier EAAI 2026 Paper Specification | Our Smart Grid Implementation Setup |
| :--- | :--- | :--- |
| **Target Application** | High-frequency industrial load monitoring & wind turbine output. | Multi-tier aggregate grid load forecasting (UCI Benchmark). |
| **SSM Backbone** | Mamba-2 state space with selective scan structured matrices. | Bidirectional Mamba formulation with forward/backward hidden states. |
| **KAN Spline Basis** | Cubic B-splines with grid adaptation over *G* = 5 knots. | Optimized B-spline parameterization regularized with *L*<sub>1</sub> knot penalties. |
| **Lookback / Horizon** | *L* = 96 &rarr; *H* = 24 (hourly load forecasting). | *L* = 28 &rarr; *H* &isin; {1, 7, 14} days multi-horizon grid dispatch. |
| **Hardware Efficiency** | Demonstrated 3.4x faster training speed than Transformer baselines. | Trained in 18.4s on RTX 3050; 42% lower peak VRAM than CLM-Former. |
| **Empirical Outcome** | Rank 1 across industrial multi-periodic test suites. | Achieved **MAE = 832,110 kW**; exceptional parameter efficiency. |

---

## Page 10: Multi-Model Benchmark & Multi-Horizon Forecast Visualizations

### 9.1 Master Benchmark Across All 12 Evaluated Models (60-Day Holdout)
| Rank | Model Architecture | Category | MAE (kW) | RMSE (kW) | MAPE (%) | Dir. Acc (%) | Inference Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **CLM-Former (Nature 2026)** | 2026 Deep SOTA | **794,845** | **1,029,915** | **23.8%** | **81.4%** | 4.2 ms |
| **2** | **Bidirectional LSTM (Bi-LSTM)** | Deep Sequence | 818,920 | 1,058,400 | 24.7% | 79.7% | 3.1 ms |
| **3** | **Temporal Convolutional (TCN)** | Deep Sequence | 825,300 | 1,066,200 | 24.9% | 78.0% | 1.8 ms |
| **4** | **Standard Stacked LSTM** | Deep Sequence | 831,450 | 1,072,100 | 25.1% | 78.0% | 2.8 ms |
| **5** | **MKJR Mamba-KAN (Elsevier 2026)** | 2026 Deep SOTA | 832,110 | 1,074,320 | 25.1% | 78.0% | 2.2 ms |
| 6 | **XGBoost Regressor** | Machine Learning | 833,410 | 1,071,280 | 25.1% | 79.7% | 1.2 ms |
| 7 | **CatBoost Regressor** | Machine Learning | 838,950 | 1,076,800 | 25.3% | 78.0% | 2.1 ms |
| 8 | **SARIMA Rolling 1-Step** | Classical Statistical | 874,120 | 1,118,450 | 25.9% | 71.2% | 14.2 ms |
| 9 | **Holt-Winters (Additive)** | Exponential Smoothing | 882,654 | 1,135,920 | 26.3% | 67.8% | 0.2 ms |
| 10 | **Seasonal Naive (Lag 7)** | Operational Baseline | 927,091 | 1,180,412 | 27.8% | 62.7% | < 0.01 ms |
| 11 | **Facebook Prophet** | Generalized Additive | 912,400 | 1,162,300 | 27.4% | 66.1% | 3.4 ms |
| 12 | **Naive (Lag 1)** | Persistence Baseline | 1,353,205 | 1,620,118 | 41.2% | 50.8% | < 0.01 ms |

### 9.2 Forecast Overlay & Multi-Horizon Decay Visualizations
![Figure 3: Model Forecasts Overlay](./figures/fig3_model_forecasts_overlay.png)
*Figure 3: 60-Day Holdout Forecast Overlay across Actual Grid Load, ARIMA flatline collapse, Holt-Winters, Seasonal Naive, XGBoost, and CLM-Former.*

![Figure 5: Multi-Horizon Error Decay](./figures/fig5_multi_horizon_decay.png)
*Figure 5: Multi-Horizon Error Accumulation (*H* = 1 to 14 days). XGBoost compounds errors (+66.4%), while CLM-Former degrades by only +5.5%.*

---

## Page 11: Operational Grid Ramping & Directional Confusion Matrices

### 10.1 Operational Problem Formulation
Grid operators must decide whether to dispatch spinning reserves or curtail thermal generation:
&Delta;*y*<sub>*t*</sub> = *y*<sub>*t*</sub> − *y*<sub>*t*−1</sub>, &emsp; Actual Class *C*<sub>*t*</sub> = **I**(&Delta;*y*<sub>*t*</sub> &ge; 0) &isin; {0 (Down-Ramp), 1 (Up-Ramp)}  
&Delta;*y&#770;*<sub>*t*</sub> = *y&#770;*<sub>*t*</sub> − *y*<sub>*t*−1</sub>, &emsp; Predicted Class *C&#770;*<sub>*t*</sub> = **I**(&Delta;*y&#770;*<sub>*t*</sub> &ge; 0) &isin; {0 (Down-Ramp), 1 (Up-Ramp)}

### 10.2 Confusion Matrices & Classification Metrics
![Figure 4: Directional Confusion Matrices](./figures/fig4_confusion_matrices.png)
*Figure 4: Directional Movement Confusion Matrices across 60 Holdout Evaluation Timestamps (Seasonal Naive, XGBoost, CLM-Former, Holt-Winters).*

| Model Architecture | True Down (TN) | False Up (FP) | False Down (FN) | True Up (TP) | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | Cohen's Kappa (&kappa;) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLM-Former (2026 SOTA)** | **21** | **6** | **5** | **27** | **81.4%** | **81.8%** | **84.4%** | **77.8%** | **0.831** | **0.624** |
| **XGBoost Regressor** | 20 | 7 | 5 | 27 | 79.7% | 79.4% | 84.4% | 74.1% | 0.818 | 0.589 |
| **Holt-Winters (Additive)** | 18 | 9 | 10 | 22 | 67.8% | 71.0% | 68.8% | 66.7% | 0.699 | 0.354 |
| **Seasonal Naive (Lag 7)** | 17 | 10 | 12 | 20 | 62.7% | 66.7% | 62.5% | 63.0% | 0.645 | 0.254 |

### 10.3 Residual Diagnostics Suite
![Figure 6: Residual Diagnostics Suite](./figures/fig6_residual_diagnostics.png)
*Figure 6: 4-Panel CLM-Former Residual Diagnostic Suite (Time series error, Gaussian fit, Autocorrelation ACF, Normal Q-Q plot).*
- **Ljung-Box Test:** *Q* = 10.42, *p* = 0.9885 &gt; 0.05 &implies; Residuals are independent white noise.
- **Q-Q Linearity:** *R*<sup>2</sup> = 0.982, confirming Gaussian error distribution.

---

## Page 12: Granular Client Clustering, Explainability & Academic Defense

### 11.1 Disaggregated 370-Meter Customer Clustering (*K*-Means Archetypes)
![Figure 7: Client Clustering & Archetypes](./figures/fig7_client_clustering.png)
*Figure 7: Disaggregated 370-Meter Customer Segmentation into 4 Archetypes & Normalized 28-Day Representative Load Signatures.*

| Archetype Tier | Customer Count | Mean Daily Load (kW) | Volatility (CV) | Weekend Ratio | Representative Behavioral Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 0: Industrial Baseload** | 42 (11.4%) | 48,210 | 0.18 | 0.96 | Continuous 24/7 manufacturing; flat predictable load; minimal weekend drop. |
| **Tier 1: Commercial / SME** | 118 (31.9%) | 14,350 | 0.42 | 0.48 | Sharp 8 AM–6 PM daytime working peaks; massive 52% reduction on weekends. |
| **Tier 2: Residential Small** | 164 (44.3%) | 3,120 | 0.65 | 1.12 | Dual morning (7–9 AM) and evening (6–10 PM) peaks; higher weekend consumption. |
| **Tier 3: Intermittent Spikers** | 46 (12.4%) | 1,840 | 1.34 | 0.82 | High-volatility sporadic equipment; non-stationary stochastic spikes. |

### 11.2 Neural Granger Causality via Input-Output Jacobian Sensitivity
![Figure 8: Neural Granger Causality](./figures/fig8_neural_granger_causality.png)
*Figure 8: Evaluation-Time Jacobian Sensitivity Matrix (|&part;y&#770; / &part;x_{t−&tau;}|) proving resonant harmonic peaks at Lags 7, 14, 21, and 28.*

### 11.3 Academic Defense & Viva Voce Strategy (High-Yield Technical Q&A)
- **Q1: Why do ARIMA and Holt-Winters look like flat straight lines in multi-step forecasting?**  
  *Answer:* Future error shocks have zero expectation (E[&epsilon;] = 0), causing MA terms to vanish after *q* steps. Stationary AR terms decay exponentially (&phi;<sup>*h*</sup> &rarr; 0), leaving only constant drift &mu; &asymp; 0.
- **Q2: Why did XGBoost outperform Deep Learning at H=1, but lose at H=14?**  
  *Answer:* At *H* = 1, yesterday's load (*y*<sub>*t*−1</sub>) is an immediate persistence oracle (*r* &gt; 0.85). However, at *H* = 14, recursive rollouts compound errors, whereas CLM-Former's multi-scale memory maintains stable accuracy.
- **Q3: How can Mamba be "Bidirectional" in time series without data leakage?**  
  *Answer:* Bidirectional scanning operates *strictly inside the 28-day historical lookback window* (*t* − 27 &rarr; *t*). It never accesses future test values.
- **Q4: What does your Ljung-Box test prove?**  
  *Answer:* The Ljung-Box test (*p* = 0.9885 &gt; 0.05) statistically confirms that CLM-Former extracted all periodic calendar information, leaving purely random Gaussian white noise.

---
*End of Technical Report — Consolidated Review 1 & Review 2 Softcopy.*
