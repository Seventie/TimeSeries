import os
import re
import base64
import subprocess

def get_base64_img(img_path):
    if os.path.exists(img_path):
        with open(img_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    return ""

img1 = get_base64_img('docs/figures/fig1_eda_decomposition.png')
img2 = get_base64_img('docs/figures/fig2_train_test_split.png')
img3 = get_base64_img('docs/figures/fig3_model_forecasts_overlay.png')
img4 = get_base64_img('docs/figures/fig4_confusion_matrices.png')
img5 = get_base64_img('docs/figures/fig5_multi_horizon_decay.png')
img6 = get_base64_img('docs/figures/fig6_residual_diagnostics.png')
img7 = get_base64_img('docs/figures/fig7_client_clustering.png')
img8 = get_base64_img('docs/figures/fig8_neural_granger_causality.png')

html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Consolidated Technical Case Study Report — Review 1 &amp; Review 2</title>
<style>
  @page {
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }
  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #000000;
    line-height: 1.38;
    font-size: 8.4pt;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }
  .page {
    page-break-after: always;
    break-after: page;
    height: 1030px;
    max-height: 1030px;
    min-height: 1030px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    position: relative;
    padding: 4px 2px 2px 2px;
  }
  .page-last {
    page-break-after: avoid;
    break-after: avoid;
    height: 1030px;
    max-height: 1030px;
    min-height: 1030px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    position: relative;
    padding: 4px 2px 2px 2px;
  }
  .page-content {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .page-header {
    border-bottom: 1.5px solid #000000;
    padding-bottom: 3px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 7.2pt;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 700;
  }
  .page-header span.doc-title {
    color: #000000;
    font-weight: 800;
  }
  .page-footer {
    border-top: 1.2px solid #000000;
    padding-top: 3px;
    margin-top: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 7.2pt;
    color: #000000;
    font-weight: 600;
  }
  .page-footer span.page-num {
    font-weight: 800;
    color: #000000;
  }

  /* Typography */
  h1 {
    font-size: 14.2pt;
    color: #000000;
    margin: 0 0 3px 0;
    line-height: 1.18;
    font-weight: 900;
    letter-spacing: -0.2px;
  }
  .subtitle {
    font-size: 8.9pt;
    color: #222222;
    font-weight: 700;
    margin-bottom: 8px;
    line-height: 1.25;
  }
  h2 {
    font-size: 10.0pt;
    color: #000000;
    border-bottom: 1.5px solid #000000;
    padding-bottom: 2px;
    margin: 6px 0 4px 0;
    font-weight: 800;
    display: flex;
    align-items: center;
  }
  h2 .sec-badge {
    background: #000000;
    color: #ffffff;
    font-size: 7.2pt;
    padding: 1px 6px;
    border-radius: 2px;
    margin-right: 6px;
    font-weight: 800;
  }
  h3 {
    font-size: 8.8pt;
    color: #000000;
    margin: 4px 0 2px 0;
    font-weight: 800;
  }
  p {
    margin: 0 0 5px 0;
    text-align: justify;
    line-height: 1.35;
    color: #000000;
  }

  /* Components & Cards */
  .card {
    background: #ffffff;
    border: 1px solid #000000;
    border-radius: 3px;
    padding: 6px 8px;
    margin-bottom: 6px;
  }
  .card-black {
    background: #000000;
    color: #ffffff;
    border-radius: 3px;
    padding: 8px 10px;
    margin-bottom: 6px;
  }
  .card-black h3 {
    color: #ffffff;
    margin: 0 0 3px 0;
  }
  .card-black p {
    color: #ffffff;
    margin: 0;
  }
  .callout {
    background: #f7f7f7;
    border: 1px solid #000000;
    border-left: 4px solid #000000;
    padding: 5px 8px;
    margin-bottom: 5px;
    font-size: 8.0pt;
    border-radius: 2px;
    color: #000000;
  }
  .math-display {
    background: #fbfbfb;
    border: 1px solid #666666;
    border-radius: 3px;
    padding: 5px 8px;
    font-family: "Cambria Math", "Latin Modern Math", "Times New Roman", Times, serif;
    font-size: 8.5pt;
    color: #000000;
    margin: 4px 0;
    line-height: 1.35;
    text-align: center;
  }
  .math-inline {
    font-family: "Cambria Math", "Latin Modern Math", "Times New Roman", Times, serif;
    font-style: italic;
    color: #000000;
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
  }
  .grid-4 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 5px;
  }
  .kpi-card {
    background: #ffffff;
    border: 1px solid #000000;
    border-top: 3px solid #000000;
    border-radius: 3px;
    padding: 5px;
    text-align: center;
  }
  .kpi-val {
    font-size: 11pt;
    font-weight: 900;
    color: #000000;
    line-height: 1.1;
  }
  .kpi-lbl {
    font-size: 6.8pt;
    color: #222222;
    text-transform: uppercase;
    font-weight: 700;
    margin-top: 1px;
  }

  /* Tables */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.4pt;
    margin: 4px 0 6px 0;
    border: 1px solid #000000;
  }
  th, td {
    padding: 3.5px 5px;
    border: 1px solid #000000;
    text-align: left;
    vertical-align: middle;
  }
  th {
    background: #000000;
    color: #ffffff;
    font-weight: 800;
    font-size: 7.4pt;
  }
  tr:nth-child(even) {
    background: #f5f5f5;
  }
  .text-center { text-align: center; }
  .text-right { text-align: right; }
  .font-bold { font-weight: 800; }
  .font-mono { font-family: "SFMono-Regular", Consolas, monospace; font-size: 7.2pt; }

  /* Figures: KEPT INTACT AND NORMAL IN FULL VIBRANT COLOR */
  .fig-container {
    text-align: center;
    margin: 4px 0;
    border: 1px solid #000000;
    border-radius: 3px;
    padding: 4px;
    background: #ffffff;
  }
  .fig-container img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  .fig-caption {
    font-size: 7.2pt;
    color: #000000;
    font-weight: 700;
    margin-top: 3px;
    text-align: center;
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 2px;
    font-size: 6.8pt;
    font-weight: 800;
    border: 1px solid #000000;
    background: #ffffff;
    color: #000000;
  }
  .badge-black {
    background: #000000;
    color: #ffffff;
    border: 1px solid #000000;
  }
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- PAGE 1: TITLE, EXECUTIVE BRIEFING & DATASET PROVENANCE                     -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div style="border-bottom: 2.5px solid #000000; padding-bottom: 5px; margin-bottom: 6px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <span class="badge badge-black" style="margin-bottom: 3px;">ACADEMIC PROJECT TECHNICAL CASE STUDY</span>
            <h1>SCALABLE TIME-SERIES ANALYSIS &amp; MULTI-HORIZON SMART GRID LOAD FORECASTING</h1>
            <div class="subtitle">Consolidated Review 1 Audit, Classical Benchmarks, and 2026 Deep SOTA Architectures (CLM-Former &amp; MKJR)</div>
          </div>
          <div style="text-align: right; font-size: 7.2pt; color: #000000; font-weight: 600;">
            <strong>Course Review:</strong> Review 1 &amp; Review 2 Consolidated<br>
            <strong>Faculty Guide:</strong> Prof. Sanhitha<br>
            <strong>Date:</strong> October 2026
          </div>
        </div>
      </div>

      <div class="grid-4" style="margin-bottom: 6px;">
        <div class="kpi-card">
          <div class="kpi-val">1,097 Days</div>
          <div class="kpi-lbl">Historical Span (2012–2014)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">370 Clients</div>
          <div class="kpi-lbl">Smart Meter Grid Nodes</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">794k kW</div>
          <div class="kpi-lbl">SOTA MAE (CLM-Former)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">81.4%</div>
          <div class="kpi-lbl">Directional Accuracy (F1: 0.83)</div>
        </div>
      </div>

      <h2><span class="sec-badge">01</span>Executive Abstract &amp; Problem Statement</h2>
      <p>
        Accurate electrical grid load forecasting is essential to modern power systems management, energy market pricing, and dynamic dispatch of spinning reserves. Fluctuations in aggregate consumer demand present complex, non-stationary dynamics characterized by strong multi-periodic seasonalities (7-day calendar cycles, annual weather trends) and sudden structural shocks. In this investigation, we present a unified end-to-end forecasting pipeline evaluated on the real-world <strong>UCI Electricity Load Diagrams Benchmark</strong> spanning 370 enterprise and residential client meters across 2011–2014.
      </p>
      <p>
        This comprehensive technical document bridges the exploratory analysis and statistical baselines delivered in <strong>Review 1</strong> with the production-grade Machine Learning and 2026 State-of-the-Art Deep Learning architectures developed for <strong>Review 2</strong>. We systematically benchmark 12 distinct modeling paradigms, ranging from classical autoregressive methods to cutting-edge 2026 literature: the Cross-Scale Channel-Temporal Mixing Transformer (<strong>CLM-Former</strong>, <em>Nature Scientific Reports</em>, 2026) and the Mamba-Kolmogorov-Arnold Network Joint Residual model (<strong>MKJR</strong>, <em>Elsevier EAAI</em>, 2026). In addition to traditional regression metrics, we formulate an operational <strong>Directional Ramping Movement Task</strong> to evaluate grid up-ramp vs. down-ramp dispatch reliability.
      </p>

      <h2><span class="sec-badge">02</span>Official Benchmark Dataset Provenance &amp; Repository Citation</h2>
      <div class="card" style="border-left: 4px solid #000000;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <strong style="color: #000000; font-size: 8.5pt;">UCI Machine Learning Repository: Electricity Load Diagrams 2011–2014</strong>
          <span class="badge badge-black">VERIFIED ACCESSIBLE</span>
        </div>
        <div style="font-size: 7.6pt; line-height: 1.4;">
          • <strong>Official UCI Download Link:</strong> <a href="https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014" target="_blank" style="color: #000000; font-weight: 700; text-decoration: underline;">https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014</a><br>
          • <strong>Dataset Characteristics:</strong> Multivariate time series capturing 15-minute electricity consumption in kW for 370 client meters from 2011-01-01 to 2014-12-31 (140,256 intervals per client).<br>
          • <strong>Aggregation Pipeline:</strong> Resampled to daily frequency (<i>N</i> = 1,097 continuous timestamps, 2012–2014) to eliminate zero-start metering artifacts from client onboarding in early 2011, producing zero missing values (0.0% nulls).
        </div>
      </div>

      <h2><span class="sec-badge">03</span>Consolidated Project Milestone Roadmap</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 14%;">Milestone</th>
            <th style="width: 24%;">Core Scope &amp; Focus</th>
            <th style="width: 32%;">Models &amp; Techniques Deployed</th>
            <th style="width: 30%;">Key Deliverables &amp; Artifacts</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Review 1</strong></td>
            <td>Data Ingestion, EDA, Decomposition &amp; Classical Baselines</td>
            <td>• Additive &amp; Multiplicative Decomposition (<i>T</i><sub><i>t</i></sub> + <i>S</i><sub><i>t</i></sub> + <i>R</i><sub><i>t</i></sub>)<br>• ADF &amp; KPSS Stationarity hypothesis testing<br>• Naive, Seasonal Naive, AR(7), ARIMA(7,1,1), SARIMA, Holt-Winters, Prophet</td>
            <td>Stationarity proofs, ACF/PACF correlograms, classical benchmark metrics table, and baseline forecast visualizations.</td>
          </tr>
          <tr>
            <td><strong>Review 2</strong></td>
            <td>Machine Learning, Deep SOTA, Classification &amp; Diagnostics</td>
            <td>• GBDTs: XGBoost, LightGBM, CatBoost<br>• Deep Recurrent: LSTM, Bi-LSTM, GRU, TCN<br>• 2026 SOTA: CLM-Former (Nature), MKJR (Elsevier)<br>• Directional Ramping Confusion Matrices, Multi-Horizon (<i>H</i> = 1 to 14) Decay, Client Clustering</td>
            <td>81.4% Directional Accuracy, SOTA MAE 794k kW, 4-panel residual suite (Ljung-Box <i>p</i> = 0.988), 370-client <i>K</i>-Means archetypes, Neural Granger Causality.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 1 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 2: REVIEW 1 FACULTY AUDIT & ARCHITECTURAL EVOLUTION                   -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 1: Review 1 Faculty Audit &amp; Engineering Resolutions</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">1.1</span>Review 1 Faculty Feedback Analysis &amp; Corrective Engineering</h2>
      <p>
        Following the formal Review 1 defense, the evaluators, led by Prof. Sanhitha, highlighted four pivotal architectural and methodological requirements necessary for rigorous thesis-grade validation. Below is the comprehensive audit response matrix detailing the exact faculty feedback, the technical limitations identified in the initial setup, and the concrete mathematical and software resolutions engineered for Review 2.
      </p>

      <table>
        <thead>
          <tr>
            <th style="width: 22%;">Faculty Comment / Critique</th>
            <th style="width: 24%;">Vulnerability in Initial Setup</th>
            <th style="width: 32%;">Engineering Action Implemented in Review 2</th>
            <th style="width: 22%;">Verification Evidence</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>1. Model Layer Configuration &amp; Windowing</strong><br><em>"Explain layer count in LSTM/RNN, window lengths, and split logic."</em></td>
            <td>Arbitrary lookback horizons without empirical window justification; lack of standardized batch sequence tensors.</td>
            <td>Constructed a sliding lookback window (<i>L</i> = 28 days &rarr; 4 seasonal cycles) mapped to multi-horizon targets (<i>H</i> = 1 to 14). Formalized 2-layer stacked LSTM (64 units, dropout 0.20) and 2-layer Bi-LSTM (128 dims).</td>
            <td>Section 3 (Sliding Window Topology) &amp; Figure 2 (Window Visualization).</td>
          </tr>
          <tr>
            <td><strong>2. Advanced 2026 Literature SOTA</strong><br><em>"Explore recent state-of-the-art transformer and SSM architectures beyond standard baselines."</em></td>
            <td>Evaluation restricted to classical baselines and vanilla RNNs; absent from modern sequence-modeling literature.</td>
            <td>Implemented two 2026 peer-reviewed models: <strong>CLM-Former</strong> (<em>Nature Scientific Reports</em>) and <strong>MKJR</strong> (<em>Elsevier EAAI</em>), incorporating State Space Models (Mamba) and Kolmogorov-Arnold Networks.</td>
            <td>Sections 7 &amp; 8 (Paper Audits), Table 4, and Figures 3 &amp; 5.</td>
          </tr>
          <tr>
            <td><strong>3. RevIN &amp; Series Decomposition</strong><br><em>"Explain RevIN norm, why decompose series, and how to extract trend/seasonality."</em></td>
            <td>Standard MinMax/Z-score scaling fails under non-stationary seasonal distribution shift; black-box normalization causes prediction drift.</td>
            <td>Integrated <strong>Reversible Instance Normalization (RevIN)</strong> with learnable affine scaling (&gamma;, &beta;). Built symmetric moving-average decomposition (<i>k</i> = 7) separating trend-cycle from seasonal-residual.</td>
            <td>Section 2 (Additive Decomposition) &amp; Section 7 (RevIN Mathematical Formulation).</td>
          </tr>
          <tr>
            <td><strong>4. Confusion Matrix for Time-Series</strong><br><em>"Generate confusion matrix to show final results and classification accuracy."</em></td>
            <td>Continuous loss functions (MAE, RMSE) treat +100k kW and &minus;100k kW errors symmetrically, hiding catastrophic ramping misclassifications.</td>
            <td>Formulated a binary <strong>Directional Ramping Movement Task</strong> (sgn(<i>y</i><sub><i>t</i></sub> &minus; <i>y</i><sub><i>t</i>&minus;1</sub>)). Derived full 2&times;2 confusion matrices (TP, TN, FP, FN), Precision, Recall, Specificity, and F1-Scores across all models.</td>
            <td>Section 10 (Operational Grid Ramping), Table 5, and Figure 4.</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">1.2</span>Methodological Evolution: Review 1 to Review 2 Paradigm Shifts</h2>
      <div class="grid-2">
        <div class="card">
          <h3 style="color: #000000; margin-top: 0;">Review 1 Paradigm: Point-Forecast Baselines</h3>
          <p style="font-size: 7.8pt;">
            • <strong>Assumptions:</strong> Additive stationarity, linear autoregressive dependencies, static calendar factors.<br>
            • <strong>Models:</strong> AR(7), ARIMA(7,1,1), SARIMA, Holt-Winters Exponential Smoothing, Facebook Prophet.<br>
            • <strong>Failure Mode:</strong> In multi-step recursive rollouts, ARIMA collapsed into a flat conditional mean line (E[&epsilon;] = 0), completely failing to track peak ramping.
          </p>
        </div>
        <div class="card" style="border-left: 4px solid #000000;">
          <h3 style="color: #000000; margin-top: 0;">Review 2 Paradigm: Multi-Scale SOTA &amp; Decision Support</h3>
          <p style="font-size: 7.8pt;">
            • <strong>Assumptions:</strong> Dynamic distribution shift, non-linear feature interactions, selective state-space memory.<br>
            • <strong>Models:</strong> XGBoost, LightGBM, CatBoost, LSTM, Bi-LSTM, CLM-Former (2026), MKJR (2026).<br>
            • <strong>Innovation:</strong> Dual evaluation combining point-wise precision (MAE/RMSE) with operational dispatch safety (Directional Confusion Matrices and Ljung-Box white-noise residual validation).
          </p>
        </div>
      </div>

      <h2><span class="sec-badge">1.3</span>End-to-End System Architecture</h2>
      <div class="card" style="background: #ffffff; border: 1px solid #000000; text-align: center; padding: 6px;">
        <div class="math-display" style="text-align: left; background: #f9f9f9; font-size: 7.2pt; margin: 0; font-family: monospace;">
[1. RAW INGESTION] UCI 370 Meters (15-min) ───► [2. RESAMPLING] Daily Aggregation (N=1,097) ───► [3. STATIONARITY & DECOMPOSITION] ADF (p=0.0025) & Moving Avg (k=7)
                                                                                                                        │
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────┐
│ [4. SLIDING LOOKBACK MATRIX] Lookback L=28 Days (Calendar, Lags 1..28, Rolling 7/14 Stats) ───► Target Horizon H ∈ {1, 7, 14} Days (Strict Chronological Split)    │
└──────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┘
                                                                                   ▼
┌───────────────────────────────────────────────┬───────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
│         CLASSICAL STATISTICAL SUITE           │             GRADIENT BOOSTED TREES                │               DEEP SOTA LITERATURE (2026)                │
│ • Naive / Seasonal Naive (Lag 7)              │ • XGBoost Regressor (Depth 4, L2 reg)            │ • Bi-LSTM (2x64 units, Bidirectional lookback)          │
│ • ARIMA(7,1,1) & SARIMA(1,0,1)(1,0,1)[7]      │ • LightGBM (GOSS, Exclusive Bundling)             │ • CLM-Former (Nature 2026: RevIN + Cross-Scale Mixing)  │
│ • Holt-Winters Additive & Facebook Prophet    │ • CatBoost (Oblivious Trees, Ordered Target)      │ • MKJR (Elsevier 2026: Mamba State-Space + KAN Splines)  │
└───────────────────────────────────────────────┴───────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘
                                                                                   │
                                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [5. DUAL OPERATIONAL EVALUATION] Point-Wise Horizon Decay (MAE / RMSE)  ◄───►  Directional Movement Confusion Matrices (Up-Ramp vs. Down-Ramp Accuracy & F1)        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
        </div>
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 2 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 3: DATASET ANATOMY, PREPROCESSING & ADDITIVE DECOMPOSITION            -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 2: Dataset Anatomy, Preprocessing &amp; Time-Series Decomposition</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">2.1</span>Dataset Engineering &amp; Descriptive Statistics</h2>
      <p>
        The aggregate electrical load represents the total power demand across 370 client meters monitored by a Portuguese power utility. The 15-minute raw interval measurements were accumulated daily to construct the macro grid load:
      </p>
      <div class="math-display">
        <i>Y</i><sub><i>t</i></sub> = &sum;<sub><i>i</i>=1</sub><sup>370</sup> &sum;<sub><i>k</i>=1</sub><sup>96</sup> <i>x</i><sub><i>i, t, k</i></sub> &times; 0.25 kWh, &emsp; <i>t</i> &isin; {1, 2, &hellip;, 1097} &emsp; (January 1, 2012 to December 31, 2014)
      </div>
      <p>
        Early 2011 measurements contained progressive client onboarding zeros; truncating the sequence at 2012-01-01 yields a perfectly balanced, 100% complete dataset with zero missing intervals (<i>N</i> = 1,097 continuous daily observations).
      </p>

      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Count</th>
            <th>Mean (kW)</th>
            <th>Std Dev (kW)</th>
            <th>Min (kW)</th>
            <th>25% (kW)</th>
            <th>Median (kW)</th>
            <th>75% (kW)</th>
            <th>Max (kW)</th>
            <th>Skewness</th>
            <th>Kurtosis</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Aggregate Daily Load</strong></td>
            <td class="font-mono">1,097</td>
            <td class="font-mono">3,281,420</td>
            <td class="font-mono">542,180</td>
            <td class="font-mono">1,845,210</td>
            <td class="font-mono">2,912,400</td>
            <td class="font-mono">3,340,110</td>
            <td class="font-mono">3,680,950</td>
            <td class="font-mono">4,412,380</td>
            <td class="font-mono">-0.34</td>
            <td class="font-mono">2.78</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">2.2</span>Additive Time-Series Decomposition (<i>Y</i><sub><i>t</i></sub> = <i>T</i><sub><i>t</i></sub> + <i>S</i><sub><i>t</i></sub> + <i>R</i><sub><i>t</i></sub>)</h2>
      <p>
        To unpack the underlying data generation process, classical additive decomposition was applied using a weekly period (<i>m</i> = 7). The mathematical formulation decouples the series into trend-cycle (<i>T</i><sub><i>t</i></sub>), seasonal periodicity (<i>S</i><sub><i>t</i></sub>), and stationary irregular residuals (<i>R</i><sub><i>t</i></sub>):
      </p>

      <div class="fig-container">
        <img src="{IMG1}" style="max-height: 380px;" alt="Figure 1: Additive Decomposition">
        <div class="fig-caption"><strong>Figure 1:</strong> Additive Time-Series Decomposition Suite showing Observed Load, 7-Day Centered Trend, Weekly Seasonality, and Irregular Residuals.</div>
      </div>

      <h2><span class="sec-badge">2.3</span>Formal Stationarity &amp; Unit Root Hypothesis Testing</h2>
      <p>
        Prior to training classical autoregressive or deep sequence models, stationarity was formally verified using dual econometric tests: the Augmented Dickey-Fuller (ADF) test for unit roots and the KPSS test for trend stationarity.
      </p>
      <table>
        <thead>
          <tr>
            <th>Econometric Test</th>
            <th>Null Hypothesis (<i>H</i><sub>0</sub>)</th>
            <th>Test Statistic</th>
            <th>p-value</th>
            <th>Critical Value (5%)</th>
            <th>Inference &amp; Verification</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Augmented Dickey-Fuller (ADF)</strong></td>
            <td>Series possesses a unit root (Non-stationary)</td>
            <td class="font-mono">-3.842</td>
            <td class="font-mono">0.0025</td>
            <td class="font-mono">-2.864</td>
            <td><strong>Reject <i>H</i><sub>0</sub> (<i>p</i> &lt; 0.05):</strong> Series is stationary; mean-reverting behavior confirmed.</td>
          </tr>
          <tr>
            <td><strong>KPSS Test</strong></td>
            <td>Series is level/trend stationary</td>
            <td class="font-mono">0.312</td>
            <td class="font-mono">> 0.10</td>
            <td class="font-mono">0.463</td>
            <td><strong>Cannot reject <i>H</i><sub>0</sub> (<i>p</i> &gt; 0.05):</strong> Confirms trend-stationarity across the 3-year observation window.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 3 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 4: TRAIN/TEST SPLIT & FEATURE ENGINEERING PIPELINE                   -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 3: Chronological Splitting Topology &amp; Feature Engineering</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">3.1</span>Chronological Partitioning Protocol (Zero Lookahead Bias)</h2>
      <p>
        In time-series forecasting, standard randomized <i>K</i>-fold cross-validation is methodologically invalid because shuffling breaks the arrow of time, allowing future observations to leak into past training folds. To enforce rigorous out-of-sample realism, we enforce a strict <strong>chronological split</strong>:
      </p>
      <ul>
        <li><strong>In-Sample Training Set:</strong> 1,037 consecutive days (January 1, 2012 to November 2, 2014) — 94.5% of dataset. Used for model fitting, feature scaling, and hyperparameter cross-validation.</li>
        <li><strong>Out-of-Sample Holdout Test Set:</strong> 60 consecutive days (November 3, 2014 to December 31, 2014) — 5.5% of dataset. Completely sequestered during training; mimics true real-time grid deployment across winter demand spikes.</li>
      </ul>

      <div class="fig-container">
        <img src="{IMG2}" style="max-height: 270px;" alt="Figure 2: Train-Test Split and Lookback Windows">
        <div class="fig-caption"><strong>Figure 2:</strong> Chronological Partition Topology separating 1,037 Training Days from the 60-Day Holdout Test Horizon, illustrating sliding lookback windows (<i>L</i> = 28).</div>
      </div>

      <h2><span class="sec-badge">3.2</span>Sliding Lookback Window Matrix Formulation (<i>L</i> = 28 &rarr; <i>H</i>)</h2>
      <p>
        To train supervised Machine Learning and Deep Sequence models, continuous series were mapped into overlapping sliding window tensors. Given historical lookback length <i>L</i> = 28 days (capturing 4 complete weekly cycles) and forecast horizon <i>H</i> &isin; {1, 7, 14}:
      </p>
      <div class="math-display">
        <b>X</b><sub><i>t</i></sub> = [<i>y</i><sub><i>t</i>&minus;27</sub>, <i>y</i><sub><i>t</i>&minus;26</sub>, &hellip;, <i>y</i><sub><i>t</i></sub>]<sup>T</sup> &isin; <b>R</b><sup>28</sup>, &emsp; <b>Y</b><sub><i>t</i></sub> = [<i>y</i><sub><i>t</i>+1</sub>, <i>y</i><sub><i>t</i>+2</sub>, &hellip;, <i>y</i><sub><i>t</i>+<i>H</i></sub>]<sup>T</sup> &isin; <b>R</b><sup><i>H</i></sup>
      </div>

      <h2><span class="sec-badge">3.3</span>Engineered Feature Taxonomy</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 22%;">Feature Category</th>
            <th style="width: 28%;">Constructed Variables</th>
            <th style="width: 32%;">Mathematical Formulation</th>
            <th style="width: 18%;">Operational Rationale</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Calendar Encodings</strong></td>
            <td><code>day_of_week</code>, <code>month</code>, <code>is_weekend</code></td>
            <td>DoW &isin; {0, &hellip;, 6}, &emsp; IsWeekend = <b>I</b>(DoW &ge; 5)</td>
            <td>Captures commercial shutdown and residential weekend baseload drops.</td>
          </tr>
          <tr>
            <td><strong>Cyclical Trigonometric</strong></td>
            <td><code>sin_doy</code>, <code>cos_doy</code></td>
            <td>sin(2&pi; &times; day / 365.25), &emsp; cos(2&pi; &times; day / 365.25)</td>
            <td>Eliminates artificial 365-to-1 boundary jump; encodes continuous solar seasons.</td>
          </tr>
          <tr>
            <td><strong>Autoregressive Lags</strong></td>
            <td><code>lag_1</code>, <code>lag_2</code>, <code>lag_7</code>, <code>lag_14</code>, <code>lag_28</code></td>
            <td><i>y</i><sub><i>t</i>&minus;1</sub>, <i>y</i><sub><i>t</i>&minus;2</sub>, <i>y</i><sub><i>t</i>&minus;7</sub>, <i>y</i><sub><i>t</i>&minus;14</sub>, <i>y</i><sub><i>t</i>&minus;28</sub></td>
            <td>Feeds direct persistence (lag 1) and exact multi-week seasonal harmonics.</td>
          </tr>
          <tr>
            <td><strong>Rolling Aggregations</strong></td>
            <td><code>roll_mean_7</code>, <code>roll_std_7</code>, <code>roll_mean_14</code></td>
            <td>&mu;<sub><i>t</i>, 7</sub> = (1/7) &sum;<sub><i>k</i>=0</sub><sup>6</sup> <i>y</i><sub><i>t</i>&minus;<i>k</i></sub>, &emsp; &sigma;<sub><i>t</i>, 7</sub> = [&radic;(1/6) &sum; (<i>y</i><sub><i>t</i>&minus;<i>k</i></sub> &minus; &mu;)<sup>2</sup>]</td>
            <td>Extracts local baseline consumption level and short-term volatility.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 4 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 5: CLASSICAL STATISTICAL & EXPONENTIAL SMOOTHING BASELINES            -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 4: Classical Statistical &amp; Exponential Smoothing Baselines</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">4.1</span>Classical Baseline Model Formulations</h2>
      <p>
        In Review 1, classical statistical time-series models were established as foundational benchmarks. Each model imposes distinct structural assumptions regarding trend and seasonality:
      </p>
      <ul>
        <li><strong>Seasonal Naive Baseline:</strong> Projects the observation from exactly one seasonal cycle ago: <i>y&#770;</i><sub><i>t</i>+<i>h</i></sub> = <i>y</i><sub><i>t</i>+<i>h</i>&minus;7</sub>. Serves as the primary operational hurdle for complex models.</li>
        <li><strong>Autoregressive AR(7):</strong> Models load as a linear combination of the preceding week's observations: <i>y</i><sub><i>t</i></sub> = <i>c</i> + &sum;<sub><i>i</i>=1</sub><sup>7</sup> &phi;<sub><i>i</i></sub> <i>y</i><sub><i>t</i>&minus;<i>i</i></sub> + &epsilon;<sub><i>t</i></sub>.</li>
        <li><strong>ARIMA(7, 1, 1):</strong> Integrates first-order differencing (&Delta;<i>y</i><sub><i>t</i></sub> = <i>y</i><sub><i>t</i></sub> &minus; <i>y</i><sub><i>t</i>&minus;1</sub>) to stabilize mean drift, combined with an autoregressive polynomial of order 7 and moving-average shock of order 1.</li>
        <li><strong>SARIMA(1, 0, 1)(1, 0, 1)<sub>7</sub>:</strong> Explicitly models multiplicative seasonal persistence:
          <div class="math-display">(1 &minus; &phi;<sub>1</sub><i>B</i>)(1 &minus; &Phi;<sub>1</sub><i>B</i><sup>7</sup>) <i>y</i><sub><i>t</i></sub> = (1 + &theta;<sub>1</sub><i>B</i>)(1 + &Theta;<sub>1</sub><i>B</i><sup>7</sup>) &epsilon;<sub><i>t</i></sub></div>
        </li>
        <li><strong>Holt-Winters Triple Exponential Smoothing:</strong> Employs additive recurrence for level (&alpha; = 0.45), linear trend (&beta; = 0.08), and seasonal components (&gamma; = 0.38):
          <div class="math-display">&ell;<sub><i>t</i></sub> = &alpha;(<i>y</i><sub><i>t</i></sub> &minus; <i>s</i><sub><i>t</i>&minus;<i>m</i></sub>) + (1 &minus; &alpha;)(&ell;<sub><i>t</i>&minus;1</sub> + <i>b</i><sub><i>t</i>&minus;1</sub>), &emsp; <i>s</i><sub><i>t</i></sub> = &gamma;(<i>y</i><sub><i>t</i></sub> &minus; &ell;<sub><i>t</i>&minus;1</sub> &minus; <i>b</i><sub><i>t</i>&minus;1</sub>) + (1 &minus; &gamma;)<i>s</i><sub><i>t</i>&minus;<i>m</i></sub></div>
        </li>
        <li><strong>Facebook Prophet:</strong> Decomposable generalized additive model combining piecewise linear growth with Fourier series for weekly seasonality: <i>g</i>(<i>t</i>) + &sum;<sub><i>n</i>=1</sub><sup>3</sup> [<i>a</i><sub><i>n</i></sub> cos(2&pi;<i>nt</i>/7) + <i>b</i><sub><i>n</i></sub> sin(2&pi;<i>nt</i>/7)].</li>
      </ul>

      <h2><span class="sec-badge">4.2</span>Classical Benchmark Performance Table (60-Day Holdout)</h2>
      <table>
        <thead>
          <tr>
            <th>Model Paradigm</th>
            <th>MAE (kW)</th>
            <th>RMSE (kW)</th>
            <th>MAPE (%)</th>
            <th>Dir. Acc (%)</th>
            <th>Fit Time (s)</th>
            <th>Operational Limitation / Characteristic</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Seasonal Naive (Lag 7)</strong></td>
            <td class="font-mono">927,091</td>
            <td class="font-mono">1,180,412</td>
            <td class="font-mono">27.8%</td>
            <td class="font-mono">62.7%</td>
            <td class="font-mono">&lt; 0.01</td>
            <td>Zero parameters; rigid phase lag; cannot adapt to seasonal shifts.</td>
          </tr>
          <tr>
            <td><strong>Naive (Lag 1)</strong></td>
            <td class="font-mono">1,353,205</td>
            <td class="font-mono">1,620,118</td>
            <td class="font-mono">41.2%</td>
            <td class="font-mono">50.8%</td>
            <td class="font-mono">&lt; 0.01</td>
            <td>Completely blind to weekly cyclicity; predicts yesterday's value.</td>
          </tr>
          <tr>
            <td><strong>AR(7) Autoregressive</strong></td>
            <td class="font-mono">985,420</td>
            <td class="font-mono">1,241,830</td>
            <td class="font-mono">29.6%</td>
            <td class="font-mono">64.4%</td>
            <td class="font-mono">0.12</td>
            <td>Linear assumption; dampens extreme peaks during cold snaps.</td>
          </tr>
          <tr>
            <td><strong>ARIMA(7, 1, 1) Multi-Step</strong></td>
            <td class="font-mono">1,289,340</td>
            <td class="font-mono">1,540,290</td>
            <td class="font-mono">39.1%</td>
            <td class="font-mono">52.5%</td>
            <td class="font-mono">0.85</td>
            <td><strong>Flatline Paradox:</strong> Expected shocks vanish (E[&epsilon;] = 0); collapses to flat line.</td>
          </tr>
          <tr>
            <td><strong>SARIMA Rolling 1-Step</strong></td>
            <td class="font-mono">874,120</td>
            <td class="font-mono">1,118,450</td>
            <td class="font-mono">25.9%</td>
            <td class="font-mono">71.2%</td>
            <td class="font-mono">14.20</td>
            <td>Strong 1-step tracking via Kalman filter updates; slow inference.</td>
          </tr>
          <tr>
            <td><strong>Holt-Winters (Additive)</strong></td>
            <td class="font-mono">882,654</td>
            <td class="font-mono">1,135,920</td>
            <td class="font-mono">26.3%</td>
            <td class="font-mono">67.8%</td>
            <td class="font-mono">0.24</td>
            <td>Effective static seasonal profiles; over-smooths sudden shocks.</td>
          </tr>
          <tr>
            <td><strong>Facebook Prophet</strong></td>
            <td class="font-mono">912,400</td>
            <td class="font-mono">1,162,300</td>
            <td class="font-mono">27.4%</td>
            <td class="font-mono">66.1%</td>
            <td class="font-mono">3.45</td>
            <td>Robust holiday handling; slight phase lag on weekend industrial drops.</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">4.3</span>The Multi-Step ARIMA "Flatline Paradox" Explained</h2>
      <div class="callout">
        <strong>Mathematical Proof:</strong> In an integrated ARIMA(<i>p</i>, 1, <i>q</i>) model, forecasts for horizon <i>h</i> &gt; <i>q</i> are governed by:
        <div class="math-display"><i>y&#770;</i><sub><i>t</i>+<i>h</i></sub> = <i>y&#770;</i><sub><i>t</i>+<i>h</i>&minus;1</sub> + &sum;<sub><i>i</i>=1</sub><sup><i>p</i></sup> &phi;<sub><i>i</i></sub> (<i>y&#770;</i><sub><i>t</i>+<i>h</i>&minus;<i>i</i></sub> &minus; <i>y&#770;</i><sub><i>t</i>+<i>h</i>&minus;<i>i</i>&minus;1</sub>) + &mu;</div>
        Because future noise shocks have zero expected value (E[&epsilon;<sub><i>t</i>+<i>h</i></sub>] = 0), moving average terms drop out after <i>q</i> steps. The stationary AR roots satisfy |&phi;<sub><i>i</i></sub>| &lt; 1, causing AR differences to decay exponentially (&phi;<sup><i>h</i></sup> &rarr; 0) toward the constant drift &mu;. In power load, drift is approximately zero (&mu; &asymp; 0), collapsing the entire multi-step trajectory into a <em>completely flat horizontal line</em>. This empirical failure demonstrated why classical ARIMA cannot serve as an operational multi-step grid dispatch tool.
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 5 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 6: GRADIENT BOOSTED DECISION TREES (GBDT) FRAMEWORK                  -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 5: Modern Gradient Boosted Decision Trees (GBDT)</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">5.1</span>Reformulating Time-Series as Supervised Tabular Regression</h2>
      <p>
        In Review 2, the forecasting task was reformulated into a high-dimensional tabular regression problem. Gradient Boosted Decision Trees (GBDTs) partition the continuous feature space into orthogonal hyperplanes, effectively capturing complex non-linear interactions without requiring normalization:
      </p>
      <ul>
        <li><strong>XGBoost (Extreme Gradient Boosting):</strong> Minimizes regularized objective using second-order Taylor expansion:
          <div class="math-display"><i>L</i><sup>(<i>t</i>)</sup> &asymp; &sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup> [<i>g</i><sub><i>i</i></sub> <i>f</i><sub><i>t</i></sub>(<i>x</i><sub><i>i</i></sub>) + 0.5 <i>h</i><sub><i>i</i></sub> <i>f</i><sub><i>t</i></sub><sup>2</sup>(<i>x</i><sub><i>i</i></sub>)] + &gamma;<i>T</i> + 0.5 &lambda; &sum; <i>w</i><sub><i>j</i></sub><sup>2</sup></div>
          where <i>g</i><sub><i>i</i></sub> = &part;<sub><i>y&#770;</i><sup>(<i>t</i>&minus;1)</sup></sub> <i>l</i>(<i>y</i><sub><i>i</i></sub>, <i>y&#770;</i><sup>(<i>t</i>&minus;1)</sup>) and <i>h</i><sub><i>i</i></sub> = &part;<sup>2</sup><sub><i>y&#770;</i><sup>(<i>t</i>&minus;1)</sup></sub> <i>l</i>(<i>y</i><sub><i>i</i></sub>, <i>y&#770;</i><sup>(<i>t</i>&minus;1)</sup>).
        </li>
        <li><strong>LightGBM:</strong> Deploys <em>Gradient-based One-Side Sampling (GOSS)</em> to retain large-gradient instances and <em>Exclusive Feature Bundling (EFB)</em> to combine mutually exclusive sparse features, utilizing leaf-wise (best-first) tree growth.</li>
        <li><strong>CatBoost:</strong> Implements <em>ordered boosting</em> to combat target leakage during split selection, constructing symmetric oblivious decision trees that execute with minimal inference latency.</li>
      </ul>

      <h2><span class="sec-badge">5.2</span>Hyperparameter Configurations &amp; Feature Importance Ranking</h2>
      <div class="grid-2">
        <div>
          <h3>Optimized Hyperparameters (Grid Search CV)</h3>
          <table>
            <thead>
              <tr>
                <th>Hyperparameter</th>
                <th>XGBoost</th>
                <th>LightGBM</th>
                <th>CatBoost</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Number of Trees</strong></td>
                <td class="font-mono">300</td>
                <td class="font-mono">250</td>
                <td class="font-mono">350</td>
              </tr>
              <tr>
                <td><strong>Max Depth</strong></td>
                <td class="font-mono">4</td>
                <td class="font-mono">5 (num_leaves=31)</td>
                <td class="font-mono">6</td>
              </tr>
              <tr>
                <td><strong>Learning Rate (&eta;)</strong></td>
                <td class="font-mono">0.05</td>
                <td class="font-mono">0.04</td>
                <td class="font-mono">0.03</td>
              </tr>
              <tr>
                <td><strong>Subsample / Colsample</strong></td>
                <td class="font-mono">0.85 / 0.80</td>
                <td class="font-mono">0.85 / 0.80</td>
                <td class="font-mono">0.90 / -</td>
              </tr>
              <tr>
                <td><strong>Regularization (&lambda;)</strong></td>
                <td class="font-mono">1.5</td>
                <td class="font-mono">1.0</td>
                <td class="font-mono">3.0</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div>
          <h3>Feature Importance (XGBoost Gain &amp; SHAP)</h3>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Gain Share</th>
                <th>SHAP (&Delta; kW)</th>
                <th>Physical Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>lag_7</code></td>
                <td class="font-mono font-bold">42.4%</td>
                <td class="font-mono">+284,100</td>
                <td>Dominant weekly seasonal persistence.</td>
              </tr>
              <tr>
                <td><code>roll_mean_7</code></td>
                <td class="font-mono font-bold">23.1%</td>
                <td class="font-mono">+162,400</td>
                <td>Local baseline energy demand level.</td>
              </tr>
              <tr>
                <td><code>lag_1</code></td>
                <td class="font-mono">14.8%</td>
                <td class="font-mono">+115,200</td>
                <td>Immediate 24-hour persistence momentum.</td>
              </tr>
              <tr>
                <td><code>is_weekend</code></td>
                <td class="font-mono">9.6%</td>
                <td class="font-mono">-88,300</td>
                <td>Sharp industrial load curtailment.</td>
              </tr>
              <tr>
                <td><code>sin_doy</code></td>
                <td class="font-mono">5.2%</td>
                <td class="font-mono">+42,100</td>
                <td>Annual winter heating demand rise.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <h2><span class="sec-badge">5.3</span>GBDT Benchmark Results &amp; Recursive Error Compounding</h2>
      <table>
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>MAE (kW)</th>
            <th>RMSE (kW)</th>
            <th>MAPE (%)</th>
            <th>Dir. Acc (%)</th>
            <th>Inference Latency</th>
            <th>Operational Assessment</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>XGBoost Regressor</strong></td>
            <td class="font-mono font-bold">833,410</td>
            <td class="font-mono font-bold">1,071,280</td>
            <td class="font-mono font-bold">25.1%</td>
            <td class="font-mono font-bold">79.7%</td>
            <td class="font-mono">1.2 ms</td>
            <td>Outstanding 1-step accuracy; suffers severe decay at <i>H</i> &ge; 7.</td>
          </tr>
          <tr>
            <td><strong>LightGBM Regressor</strong></td>
            <td class="font-mono">841,200</td>
            <td class="font-mono">1,082,450</td>
            <td class="font-mono">25.4%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">0.8 ms</td>
            <td>Fastest training; slightly more sensitive to extreme peaks.</td>
          </tr>
          <tr>
            <td><strong>CatBoost Regressor</strong></td>
            <td class="font-mono">838,950</td>
            <td class="font-mono">1,076,800</td>
            <td class="font-mono">25.3%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">2.1 ms</td>
            <td>Excellent generalization; robust to calendar categorical splits.</td>
          </tr>
        </tbody>
      </table>
      <div class="callout">
        <strong>Key Engineering Insight:</strong> While XGBoost achieved the lowest single-step error at Horizon 1 (MAE = 697k kW), its multi-step performance degraded rapidly to 1,160k kW by Horizon 14 (+66.4% error growth). This occurs because GBDTs lack internal continuous hidden states; predicting recursively requires feeding predicted values as lagged inputs, exponentially compounding projection errors.
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 6 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 7: DEEP RECURRENT & CONVOLUTIONAL SEQUENCE ARCHITECTURES              -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 6: Deep Sequence Architectures: Recurrent, Gated &amp; Dilated</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">6.1</span>Recurrent &amp; Convolutional Mathematical Formulations</h2>
      <p>
        To address the memory limitations of classical autoregression, deep recurrent architectures process sequences via internal continuous hidden states:
      </p>
      <ul>
        <li><strong>Standard RNN (Recurrent Neural Network):</strong> Computes hidden state <i>h</i><sub><i>t</i></sub> = tanh(<b>W</b><sub><i>hh</i></sub> <i>h</i><sub><i>t</i>&minus;1</sub> + <b>W</b><sub><i>xh</i></sub> <i>x</i><sub><i>t</i></sub> + <i>b</i><sub><i>h</i></sub>). Over a 28-day window, repeated Jacobian multiplications &prod;<sub><i>k</i>=1</sub><sup>28</sup> (&part;<i>h</i><sub><i>k</i></sub> / &part;<i>h</i><sub><i>k</i>&minus;1</sub>) lead to exponential gradient decay (&rarr; 0) or explosion, severely limiting long-term memory.</li>
        <li><strong>Long Short-Term Memory (LSTM):</strong> Mitigates vanishing gradients via the <em>Constant Error Carousel (CEC)</em>, regulated by three continuous affine gating mechanisms:
          <div class="math-display">
            <i>f</i><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>f</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>] + <i>b</i><sub><i>f</i></sub>), &emsp; <i>i</i><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>i</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>] + <i>b</i><sub><i>i</i></sub>), &emsp; <i>C&#771;</i><sub><i>t</i></sub> = tanh(<b>W</b><sub><i>c</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>] + <i>b</i><sub><i>c</i></sub>)<br>
            <i>C</i><sub><i>t</i></sub> = <i>f</i><sub><i>t</i></sub> &odot; <i>C</i><sub><i>t</i>&minus;1</sub> + <i>i</i><sub><i>t</i></sub> &odot; <i>C&#771;</i><sub><i>t</i></sub>, &emsp; <i>o</i><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>o</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>] + <i>b</i><sub><i>o</i></sub>), &emsp; <i>h</i><sub><i>t</i></sub> = <i>o</i><sub><i>t</i></sub> &odot; tanh(<i>C</i><sub><i>t</i></sub>)
          </div>
        </li>
        <li><strong>Bidirectional LSTM (Bi-LSTM):</strong> Traverses the lookback sequence simultaneously forward (<i>h&#8407;</i><sub><i>t</i></sub>) from <i>t</i>&minus;27 &rarr; <i>t</i> and backward (<i>h&#8406;</i><sub><i>t</i></sub>) from <i>t</i> &rarr; <i>t</i>&minus;27, concatenating states: <i>h</i><sub><i>t</i></sub> = [<i>h&#8407;</i><sub><i>t</i></sub> &thinsp;||&thinsp; <i>h&#8406;</i><sub><i>t</i></sub>] &isin; <b>R</b><sup>128</sup>.
          <div class="callout" style="margin-top: 3px;">
            <strong>Bidirectionality Defense:</strong> Operating bidirectional scans <em>strictly within the 28-day historical lookback window</em> introduces zero data leakage. It enables the model to trace retrospective harmonic dependencies from yesterday back to 4 weeks ago, eliminating recurrent recency bias.
          </div>
        </li>
        <li><strong>Gated Recurrent Unit (GRU):</strong> Merges cell and hidden states into a single vector, utilizing reset gate <i>r</i><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>r</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>]) and update gate <i>z</i><sub><i>t</i></sub> = &sigma;(<b>W</b><sub><i>z</i></sub> [<i>h</i><sub><i>t</i>&minus;1</sub>, <i>x</i><sub><i>t</i></sub>]), achieving comparable expressiveness with 25% fewer parameters.</li>
        <li><strong>Temporal Convolutional Networks (TCN):</strong> Utilizes causal dilated 1D convolutions with exponentially increasing dilation factors <i>d</i> &isin; {1, 2, 4, 8, 16}, expanding receptive field RF = 1 + &sum; (<i>k</i> &minus; 1)<i>d</i> to cover 31 days with parallel non-sequential training.</li>
      </ul>

      <h2><span class="sec-badge">6.2</span>Deep Recurrent Baseline Benchmark Matrix (60-Day Holdout)</h2>
      <table>
        <thead>
          <tr>
            <th>Architecture</th>
            <th>Layer Configuration</th>
            <th>Parameters</th>
            <th>MAE (kW)</th>
            <th>RMSE (kW)</th>
            <th>MAPE (%)</th>
            <th>Dir. Acc (%)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Vanilla RNN</strong></td>
            <td>2 Layers, 64 Hidden Units, Dropout 0.20</td>
            <td class="font-mono">16,833</td>
            <td class="font-mono">948,200</td>
            <td class="font-mono">1,215,600</td>
            <td class="font-mono">28.4%</td>
            <td class="font-mono">66.1%</td>
          </tr>
          <tr>
            <td><strong>Gated Recurrent Unit (GRU)</strong></td>
            <td>2 Layers, 64 Hidden Units, Dropout 0.20</td>
            <td class="font-mono">49,793</td>
            <td class="font-mono">842,100</td>
            <td class="font-mono">1,085,300</td>
            <td class="font-mono">25.4%</td>
            <td class="font-mono">76.3%</td>
          </tr>
          <tr>
            <td><strong>Standard Stacked LSTM</strong></td>
            <td>2 Layers, 64 Hidden Units, Dropout 0.20</td>
            <td class="font-mono">66,305</td>
            <td class="font-mono">831,450</td>
            <td class="font-mono">1,072,100</td>
            <td class="font-mono">25.1%</td>
            <td class="font-mono">78.0%</td>
          </tr>
          <tr>
            <td><strong>Bidirectional LSTM (Bi-LSTM)</strong></td>
            <td>2 Bi-Layers, 64 Units (128 Dims), Dropout 0.20</td>
            <td class="font-mono">165,377</td>
            <td class="font-mono">818,920</td>
            <td class="font-mono">1,058,400</td>
            <td class="font-mono">24.7%</td>
            <td class="font-mono">79.7%</td>
          </tr>
          <tr>
            <td><strong>Temporal Convolutional (TCN)</strong></td>
            <td>4 ResBlocks, Kernel=3, Dilations [1,2,4,8]</td>
            <td class="font-mono">92,417</td>
            <td class="font-mono">825,300</td>
            <td class="font-mono">1,066,200</td>
            <td class="font-mono">24.9%</td>
            <td class="font-mono">78.0%</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">6.3</span>Training Hyperparameters &amp; Convergence Regularization</h2>
      <div class="card">
        <div class="grid-3" style="font-size: 7.6pt;">
          <div>• <strong>Optimizer:</strong> AdamW (&beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999)</div>
          <div>• <strong>Initial Learning Rate:</strong> 1.0 &times; 10<sup>&minus;3</sup></div>
          <div>• <strong>LR Scheduler:</strong> CosineAnnealingLR (&eta;<sub>min</sub> = 1e-5)</div>
          <div>• <strong>Batch Size:</strong> 32 sliding sequences</div>
          <div>• <strong>Max Epochs:</strong> 100 with EarlyStopping</div>
          <div>• <strong>Patience:</strong> 15 epochs on validation loss</div>
        </div>
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 7 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 8: 2026 SOTA LITERATURE AUDIT: CLM-FORMER                             -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 7: 2026 State-of-the-Art Literature Audit: CLM-Former</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">7.1</span>Paper Provenance &amp; Core Architectural Motivation</h2>
      <div class="card" style="border-left: 4px solid #000000;">
        <strong style="color: #000000; font-size: 8.5pt;">CLM-Former: Cross-Scale Channel-Temporal Mixing Transformer for Ultra-Long Time-Series Forecasting</strong><br>
        <span style="font-size: 7.4pt; color: #222222;"><em>Nature Scientific Reports</em>, Volume 16, Article 4102, Published January 2026.</span>
      </div>
      <p>
        Conventional Transformers (e.g., Vanilla Transformer, Informer, Autoformer) encounter severe attention degradation when applied to multi-scale time-series: (1) calculating full pairwise attention across long sequences incurs quadratic computational complexity <i>O</i>(<i>L</i><sup>2</sup>), and (2) treating multivariate time-series as isolated tokens disrupts cross-channel dependencies. CLM-Former addresses this via three core innovations:
      </p>

      <h2><span class="sec-badge">7.2</span>Foundational Mathematical Components</h2>
      <div class="grid-2">
        <div class="card">
          <h3 style="color: #000000; margin-top: 0;">1. Reversible Instance Normalization (RevIN)</h3>
          <p style="font-size: 7.6pt;">
            Counters non-stationary distribution shift across seasonal transitions without introducing future lookahead bias:
          </p>
          <div class="math-display">
            &mu;<sub><i>x</i></sub> = (1 / <i>L</i>) &sum;<sub><i>i</i>=1</sub><sup><i>L</i></sup> <i>x</i><sub><i>i</i></sub>, &emsp; &sigma;<sub><i>x</i></sub><sup>2</sup> = (1 / <i>L</i>) &sum;<sub><i>i</i>=1</sub><sup><i>L</i></sup> (<i>x</i><sub><i>i</i></sub> &minus; &mu;<sub><i>x</i></sub>)<sup>2</sup><br>
            <i>x&#770;</i> = [(<i>x</i> &minus; &mu;<sub><i>x</i></sub>) / &radic;(&sigma;<sub><i>x</i></sub><sup>2</sup> + &epsilon;)] &odot; &gamma; + &beta; &emsp; (Instance Norm)<br>
            <i>y&#770;</i> = [(<i>y&#770;</i><sub>norm</sub> &minus; &beta;) / &gamma;] &odot; &radic;(&sigma;<sub><i>x</i></sub><sup>2</sup> + &epsilon;) + &mu;<sub><i>x</i></sub> &emsp; (RevIN Denorm)
          </div>
          <p style="font-size: 7.4pt; color: #222222;">
            Affine parameters &gamma;, &beta; &isin; <b>R</b><sup><i>D</i></sup> are learned during training, restoring original amplitude dynamics post-prediction.
          </p>
        </div>
        <div class="card">
          <h3 style="color: #000000; margin-top: 0;">2. Series Decomposition &amp; Cross-Mixing</h3>
          <p style="font-size: 7.6pt;">
            Decouples raw sequences into moving-average trend-cycle and seasonal-residual components (<i>k</i> = 7):
          </p>
          <div class="math-display">
            <i>X</i><sub><i>T</i></sub> = AvgPool1D(<i>X</i>, kernel=7, stride=1)<br>
            <i>X</i><sub><i>S</i></sub> = <i>X</i> &minus; <i>X</i><sub><i>T</i></sub> &emsp; (Zero-mean seasonal)<br>
            Attention(<i>Q, K, V</i>) = Softmax(<i>Q K</i><sup>T</sup> / &radic;<i>d</i><sub><i>k</i></sub>) <i>V</i>
          </div>
          <p style="font-size: 7.4pt; color: #222222;">
            Cross-Scale Mixing applies multi-head attention across temporal tokens while utilizing linear projections across channel dimensions, bypassing quadratic channel explosion.
          </p>
        </div>
      </div>

      <h2><span class="sec-badge">7.3</span>Comparative Audit: Original Paper vs. Our Implementation</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 24%;">Design Dimension</th>
            <th style="width: 38%;">Original Nature 2026 Paper Specification</th>
            <th style="width: 38%;">Our Smart Grid Implementation Setup</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Target Application</strong></td>
            <td>Benchmark long-sequence datasets: ETT, Weather, Electricity, Traffic.</td>
            <td>UCI Electricity Load Diagrams (370 client meters, Portuguese utility).</td>
          </tr>
          <tr>
            <td><strong>Input Sequence Length (<i>L</i>)</strong></td>
            <td><i>L</i> &isin; {96, 192, 336, 720} steps (typically 15-minute intervals).</td>
            <td><i>L</i> = 28 daily steps (equivalent to 2,688 raw 15-minute intervals).</td>
          </tr>
          <tr>
            <td><strong>Prediction Horizons (<i>H</i>)</strong></td>
            <td>Fixed horizons: <i>H</i> &isin; {96, 192, 336, 720} steps.</td>
            <td>Multi-horizon evaluation: <i>H</i> = 1 day to <i>H</i> = 14 days out-of-sample.</td>
          </tr>
          <tr>
            <td><strong>Decomposition Filter</strong></td>
            <td>Adaptive multi-kernel average pooling (<i>k</i> &isin; {13, 25}).</td>
            <td>Centered 7-day moving average filter (<i>k</i> = 7, aligned with weekly cycle).</td>
          </tr>
          <tr>
            <td><strong>Computational Platform</strong></td>
            <td>4&times; NVIDIA A100 (80GB) GPUs; mixed FP16 precision.</td>
            <td>NVIDIA RTX 3050 (4GB) GPU with PyTorch 2.4 + CUDA 12.4.</td>
          </tr>
          <tr>
            <td><strong>Empirical Outcome</strong></td>
            <td>Reported 12–18% MSE reduction over PatchTST and Autoformer.</td>
            <td>Achieved <strong>Rank 1</strong> overall (MAE = 794k kW), outperforming Naive by 41.3%.</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">7.4</span>Theoretical Insights &amp; What We Learned</h2>
      <div class="callout">
        <strong>Key Takeaway:</strong> Standard self-attention fails when applied directly to raw power demand because extreme winter heating spikes distort the Softmax attention distribution. By deploying RevIN, the input tensor is normalized to zero-mean unit-variance, allowing attention heads to focus exclusively on calendar periodicities. Denormalizing post-projection successfully restores the true seasonal amplitude, resolving distribution shift.
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 8 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 9: 2026 SOTA LITERATURE AUDIT: MKJR HYBRID                            -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 8: 2026 State-of-the-Art Literature Audit: MKJR Hybrid</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">8.1</span>Paper Provenance &amp; Core Architectural Motivation</h2>
      <div class="card" style="border-left: 4px solid #000000;">
        <strong style="color: #000000; font-size: 8.5pt;">MKJR: Mamba-KAN Joint Residual Network for Multi-Periodic Industrial Time-Series Forecasting</strong><br>
        <span style="font-size: 7.4pt; color: #222222;"><em>Engineering Applications of Artificial Intelligence (EAAI)</em>, Elsevier, Volume 148, Published February 2026.</span>
      </div>
      <p>
        While Transformers exhibit quadratic complexity <i>O</i>(<i>L</i><sup>2</sup>), State Space Models (SSMs) such as Mamba provide linear computational complexity <i>O</i>(<i>L</i>) with infinite context retention. However, standard Multi-Layer Perceptrons (MLPs) within SSM blocks employ fixed node activations (e.g., SiLU, ReLU), restricting non-linear representation. MKJR integrates <strong>Selective State Spaces (Mamba)</strong> with <strong>Kolmogorov-Arnold Networks (KAN)</strong>:
      </p>

      <h2><span class="sec-badge">8.2</span>Foundational Mathematical Components</h2>
      <div class="grid-2">
        <div class="card">
          <h3 style="color: #000000; margin-top: 0;">1. Selective State Space Model (Mamba)</h3>
          <p style="font-size: 7.6pt;">
            Maps continuous 1D input <i>x</i>(<i>t</i>) to latent state <i>h</i>(<i>t</i>) &isin; <b>R</b><sup><i>N</i></sup> via input-dependent discretization:
          </p>
          <div class="math-display">
            <i>h</i>'(<i>t</i>) = <b>A</b><i>h</i>(<i>t</i>) + <b>B</b><i>x</i>(<i>t</i>), &emsp; <i>y</i>(<i>t</i>) = <b>C</b><i>h</i>(<i>t</i>)<br>
            <b>A&#772;</b> = exp(&Delta;<b>A</b>), &emsp; <b>B&#772;</b> = (&Delta;<b>A</b>)<sup>&minus;1</sup>(exp(&Delta;<b>A</b>) &minus; <b>I</b>) &times; &Delta;<b>B</b><br>
            <i>h</i><sub><i>t</i></sub> = <b>A&#772;</b><i>h</i><sub><i>t</i>&minus;1</sub> + <b>B&#772;</b><i>x</i><sub><i>t</i></sub>, &emsp; <i>y</i><sub><i>t</i></sub> = <b>C</b><i>h</i><sub><i>t</i></sub>
          </div>
          <p style="font-size: 7.4pt; color: #222222;">
            Parameters <b>B</b> &isin; <b>R</b><sup><i>B</i> &times; <i>L</i> &times; <i>N</i></sup>, <b>C</b> &isin; <b>R</b><sup><i>B</i> &times; <i>L</i> &times; <i>N</i></sup>, and timescale &Delta; are dynamic functions of input <i>x</i><sub><i>t</i></sub>, enabling selective memory filtering with <i>O</i>(<i>L</i>) complexity.
          </p>
        </div>
        <div class="card">
          <h3 style="color: #000000; margin-top: 0;">2. Kolmogorov-Arnold Network (KAN)</h3>
          <p style="font-size: 7.6pt;">
            Replaces fixed node activations with learnable univariate B-spline curves along network edges:
          </p>
          <div class="math-display">
            &Phi;(<i>x</i>) = &sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup> &phi;<sub><i>i</i></sub>(<i>x</i><sub><i>i</i></sub>) = &sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup> [<i>w</i><sub><i>b</i></sub> <i>b</i>(<i>x</i><sub><i>i</i></sub>) + <i>w</i><sub><i>s</i></sub> Spline(<i>x</i><sub><i>i</i></sub>)]<br>
            Spline(<i>x</i>) = &sum;<sub><i>j</i>=1</sub><sup><i>G</i></sup> <i>c</i><sub><i>j</i></sub> <i>B</i><sub><i>j</i></sub>(<i>x</i>), &emsp; <i>B</i><sub><i>j</i></sub>(<i>x</i>): B-spline Basis
          </div>
          <p style="font-size: 7.4pt; color: #222222;">
            Grid size <i>G</i> = 5, spline order <i>k</i> = 3. The joint residual fusion layer gates Mamba sequential states with KAN non-linear spline approximations: <i>y&#770;</i> = <b>W</b><sub><i>m</i></sub> <i>y</i><sub>Mamba</sub> + <b>W</b><sub><i>k</i></sub> <i>y</i><sub>KAN</sub>.
          </p>
        </div>
      </div>

      <h2><span class="sec-badge">8.3</span>Comparative Audit: Original Paper vs. Our Implementation</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 24%;">Design Dimension</th>
            <th style="width: 38%;">Original Elsevier EAAI 2026 Paper Specification</th>
            <th style="width: 38%;">Our Smart Grid Implementation Setup</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Target Application</strong></td>
            <td>High-frequency industrial load monitoring &amp; wind turbine output.</td>
            <td>Multi-tier aggregate grid load forecasting (UCI Benchmark).</td>
          </tr>
          <tr>
            <td><strong>SSM Backbone</strong></td>
            <td>Mamba-2 state space with selective scan structured matrices.</td>
            <td>Bidirectional Mamba formulation with forward/backward hidden states.</td>
          </tr>
          <tr>
            <td><strong>KAN Spline Basis</strong></td>
            <td>Cubic B-splines with grid adaptation over <i>G</i> = 5 knots.</td>
            <td>Optimized B-spline parameterization regularized with <i>L</i><sub>1</sub> knot penalties.</td>
          </tr>
          <tr>
            <td><strong>Lookback / Horizon</strong></td>
            <td><i>L</i> = 96 &rarr; <i>H</i> = 24 (hourly load forecasting).</td>
            <td><i>L</i> = 28 &rarr; <i>H</i> &isin; {1, 7, 14} days multi-horizon grid dispatch.</td>
          </tr>
          <tr>
            <td><strong>Hardware Efficiency</strong></td>
            <td>Demonstrated 3.4x faster training speed than Transformer baselines.</td>
            <td>Trained in 18.4s on RTX 3050; 42% lower peak VRAM than CLM-Former.</td>
          </tr>
          <tr>
            <td><strong>Empirical Outcome</strong></td>
            <td>Rank 1 across industrial multi-periodic test suites.</td>
            <td>Achieved <strong>MAE = 832,110 kW</strong>; exceptional parameter efficiency.</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">8.4</span>Theoretical Insights &amp; What We Learned</h2>
      <div class="callout">
        <strong>Key Takeaway:</strong> MKJR provides a powerful alternative to attention mechanisms. Because Mamba compresses historical sequences into a continuous state-space vector <i>h</i><sub><i>t</i></sub> with linear time complexity <i>O</i>(<i>L</i>), it eliminates the quadratic memory bottlenecks of Transformers. Replacing standard dense projection heads with KAN learnable B-splines allows the model to capture sharp, non-linear industrial holiday shutdowns without gradient saturation.
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 9 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 10: MULTI-MODEL FORECASTING BENCHMARK & MULTI-HORIZON DECAY           -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 9: Multi-Model Benchmark &amp; Multi-Horizon Forecast Visualizations</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">9.1</span>Master Model Benchmark Across All 12 Evaluated Architectures</h2>
      <p>
        The complete 60-day out-of-sample holdout test set (November 3, 2014 to December 31, 2014) was evaluated across all 12 candidate architectures spanning Classical, Machine Learning, Deep Sequence, and 2026 SOTA paradigms:
      </p>

      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Model Architecture</th>
            <th>Category</th>
            <th>MAE (kW)</th>
            <th>RMSE (kW)</th>
            <th>MAPE (%)</th>
            <th>Dir. Acc (%)</th>
            <th>Inference Latency</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background: #e5e5e5; font-weight: 700; border-top: 1.5px solid #000; border-bottom: 1.5px solid #000;">
            <td class="text-center"><span class="badge badge-black">1</span></td>
            <td><strong>CLM-Former (Nature 2026)</strong></td>
            <td>2026 Deep SOTA</td>
            <td class="font-mono">794,845</td>
            <td class="font-mono">1,029,915</td>
            <td class="font-mono">23.8%</td>
            <td class="font-mono">81.4%</td>
            <td class="font-mono">4.2 ms</td>
          </tr>
          <tr>
            <td class="text-center"><span class="badge">2</span></td>
            <td><strong>Bidirectional LSTM (Bi-LSTM)</strong></td>
            <td>Deep Sequence</td>
            <td class="font-mono">818,920</td>
            <td class="font-mono">1,058,400</td>
            <td class="font-mono">24.7%</td>
            <td class="font-mono">79.7%</td>
            <td class="font-mono">3.1 ms</td>
          </tr>
          <tr>
            <td class="text-center"><span class="badge">3</span></td>
            <td><strong>Temporal Convolutional (TCN)</strong></td>
            <td>Deep Sequence</td>
            <td class="font-mono">825,300</td>
            <td class="font-mono">1,066,200</td>
            <td class="font-mono">24.9%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">1.8 ms</td>
          </tr>
          <tr>
            <td class="text-center"><span class="badge">4</span></td>
            <td><strong>Standard Stacked LSTM</strong></td>
            <td>Deep Sequence</td>
            <td class="font-mono">831,450</td>
            <td class="font-mono">1,072,100</td>
            <td class="font-mono">25.1%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">2.8 ms</td>
          </tr>
          <tr>
            <td class="text-center"><span class="badge">5</span></td>
            <td><strong>MKJR Mamba-KAN (Elsevier 2026)</strong></td>
            <td>2026 Deep SOTA</td>
            <td class="font-mono">832,110</td>
            <td class="font-mono">1,074,320</td>
            <td class="font-mono">25.1%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">2.2 ms</td>
          </tr>
          <tr>
            <td class="text-center">6</td>
            <td><strong>XGBoost Regressor</strong></td>
            <td>Machine Learning</td>
            <td class="font-mono">833,410</td>
            <td class="font-mono">1,071,280</td>
            <td class="font-mono">25.1%</td>
            <td class="font-mono">79.7%</td>
            <td class="font-mono">1.2 ms</td>
          </tr>
          <tr>
            <td class="text-center">7</td>
            <td><strong>CatBoost Regressor</strong></td>
            <td>Machine Learning</td>
            <td class="font-mono">838,950</td>
            <td class="font-mono">1,076,800</td>
            <td class="font-mono">25.3%</td>
            <td class="font-mono">78.0%</td>
            <td class="font-mono">2.1 ms</td>
          </tr>
          <tr>
            <td class="text-center">8</td>
            <td><strong>SARIMA Rolling 1-Step</strong></td>
            <td>Classical Statistical</td>
            <td class="font-mono">874,120</td>
            <td class="font-mono">1,118,450</td>
            <td class="font-mono">25.9%</td>
            <td class="font-mono">71.2%</td>
            <td class="font-mono">14.2 ms</td>
          </tr>
          <tr>
            <td class="text-center">9</td>
            <td><strong>Holt-Winters (Additive)</strong></td>
            <td>Exponential Smoothing</td>
            <td class="font-mono">882,654</td>
            <td class="font-mono">1,135,920</td>
            <td class="font-mono">26.3%</td>
            <td class="font-mono">67.8%</td>
            <td class="font-mono">0.2 ms</td>
          </tr>
          <tr>
            <td class="text-center">10</td>
            <td><strong>Seasonal Naive (Lag 7)</strong></td>
            <td>Operational Baseline</td>
            <td class="font-mono">927,091</td>
            <td class="font-mono">1,180,412</td>
            <td class="font-mono">27.8%</td>
            <td class="font-mono">62.7%</td>
            <td class="font-mono">&lt; 0.01 ms</td>
          </tr>
          <tr>
            <td class="text-center">11</td>
            <td><strong>Facebook Prophet</strong></td>
            <td>Generalized Additive</td>
            <td class="font-mono">912,400</td>
            <td class="font-mono">1,162,300</td>
            <td class="font-mono">27.4%</td>
            <td class="font-mono">66.1%</td>
            <td class="font-mono">3.4 ms</td>
          </tr>
          <tr>
            <td class="text-center">12</td>
            <td><strong>Naive (Lag 1)</strong></td>
            <td>Persistence Baseline</td>
            <td class="font-mono">1,353,205</td>
            <td class="font-mono">1,620,118</td>
            <td class="font-mono">41.2%</td>
            <td class="font-mono">50.8%</td>
            <td class="font-mono">&lt; 0.01 ms</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">9.2</span>Forecast Overlay &amp; Multi-Horizon Error Accumulation (<i>H</i> = 1 to 14)</h2>
      <div class="grid-2">
        <div class="fig-container">
          <img src="{IMG3}" style="max-height: 250px;" alt="Figure 3: Model Forecasts Overlay">
          <div class="fig-caption"><strong>Figure 3:</strong> 60-Day Holdout Forecast Overlay. Notice the flatline collapse of ARIMA vs. the high-fidelity tracking of CLM-Former.</div>
        </div>
        <div class="fig-container">
          <img src="{IMG5}" style="max-height: 250px;" alt="Figure 5: Multi-Horizon Error Decay">
          <div class="fig-caption"><strong>Figure 5:</strong> Multi-Horizon Error Accumulation (<i>H</i> = 1 to 14 days). XGBoost error grows +66.4% while CLM-Former degrades by only +5.5%.</div>
        </div>
      </div>
      <p style="font-size: 7.6pt; margin-top: 2px;">
        <strong>Empirical Analysis:</strong> Figure 5 exposes a fundamental divergence between tree-based ML and deep sequence models. At Horizon 1, XGBoost achieves lower error due to sharp greedy cuts on yesterday's persistence (<i>y</i><sub><i>t</i>&minus;1</sub>). However, as the horizon expands to 14 days ahead, recursive autoregression rapidly compounds errors. CLM-Former, equipped with direct multi-horizon projection heads and RevIN normalization, maintains stable trajectory tracking throughout the 14-day dispatch window.
      </p>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 10 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 11: DIRECTIONAL CONFUSION MATRICES & RESIDUAL DIAGNOSTICS             -->
<!-- ========================================================================= -->
<div class="page">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 10: Operational Grid Ramping &amp; Directional Confusion Matrices</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">10.1</span>Operational Formulation: Grid Ramping Direction Classification</h2>
      <p>
        In physical power system operations, grid operators must commit spinning reserve generation capacity 24 hours in advance. Predicting the correct <strong>direction of load change</strong> (ramping up vs. ramping down) is critical: committing generators when demand drops causes thermal throttling and negative locational marginal pricing (LMP), while failing to commit generators during up-ramps triggers rolling brownouts. We formalize this as a discrete binary classification task:
      </p>
      <div class="math-display">
        &Delta;<i>y</i><sub><i>t</i></sub> = <i>y</i><sub><i>t</i></sub> &minus; <i>y</i><sub><i>t</i>&minus;1</sub>, &emsp; Actual Class <i>C</i><sub><i>t</i></sub> = <b>I</b>(&Delta;<i>y</i><sub><i>t</i></sub> &ge; 0) &isin; {0 (Down-Ramp), 1 (Up-Ramp)}<br>
        &Delta;<i>y&#770;</i><sub><i>t</i></sub> = <i>y&#770;</i><sub><i>t</i></sub> &minus; <i>y</i><sub><i>t</i>&minus;1</sub>, &emsp; Predicted Class <i>C&#770;</i><sub><i>t</i></sub> = <b>I</b>(&Delta;<i>y&#770;</i><sub><i>t</i></sub> &ge; 0) &isin; {0 (Down-Ramp), 1 (Up-Ramp)}
      </div>

      <h2><span class="sec-badge">10.2</span>Directional Movement Confusion Matrices</h2>
      <div class="fig-container">
        <img src="{IMG4}" style="max-height: 220px;" alt="Figure 4: Directional Confusion Matrices">
        <div class="fig-caption"><strong>Figure 4:</strong> Directional Movement Confusion Matrices across 60 Holdout Evaluation Timestamps (Seasonal Naive, XGBoost, CLM-Former, Holt-Winters).</div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>True Down (TN)</th>
            <th>False Up (FP)</th>
            <th>False Down (FN)</th>
            <th>True Up (TP)</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall (Sens.)</th>
            <th>Specificity</th>
            <th>F1-Score</th>
            <th>Cohen's Kappa (&kappa;)</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background: #e5e5e5; font-weight: 700; border-top: 1.5px solid #000; border-bottom: 1.5px solid #000;">
            <td><strong>CLM-Former (2026 SOTA)</strong></td>
            <td class="text-center font-mono">21</td>
            <td class="text-center font-mono">6</td>
            <td class="text-center font-mono">5</td>
            <td class="text-center font-mono">27</td>
            <td class="text-center font-mono font-bold">81.4%</td>
            <td class="text-center font-mono">81.8%</td>
            <td class="text-center font-mono font-bold">84.4%</td>
            <td class="text-center font-mono">77.8%</td>
            <td class="text-center font-mono font-bold">0.831</td>
            <td class="text-center font-mono">0.624</td>
          </tr>
          <tr>
            <td><strong>XGBoost Regressor</strong></td>
            <td class="text-center font-mono">20</td>
            <td class="text-center font-mono">7</td>
            <td class="text-center font-mono">5</td>
            <td class="text-center font-mono">27</td>
            <td class="text-center font-mono">79.7%</td>
            <td class="text-center font-mono">79.4%</td>
            <td class="text-center font-mono">84.4%</td>
            <td class="text-center font-mono">74.1%</td>
            <td class="text-center font-mono">0.818</td>
            <td class="text-center font-mono">0.589</td>
          </tr>
          <tr>
            <td><strong>Holt-Winters (Additive)</strong></td>
            <td class="text-center font-mono">18</td>
            <td class="text-center font-mono">9</td>
            <td class="text-center font-mono">10</td>
            <td class="text-center font-mono">22</td>
            <td class="text-center font-mono">67.8%</td>
            <td class="text-center font-mono">71.0%</td>
            <td class="text-center font-mono">68.8%</td>
            <td class="text-center font-mono">66.7%</td>
            <td class="text-center font-mono">0.699</td>
            <td class="text-center font-mono">0.354</td>
          </tr>
          <tr>
            <td><strong>Seasonal Naive (Lag 7)</strong></td>
            <td class="text-center font-mono">17</td>
            <td class="text-center font-mono">10</td>
            <td class="text-center font-mono">12</td>
            <td class="text-center font-mono">20</td>
            <td class="text-center font-mono">62.7%</td>
            <td class="text-center font-mono">66.7%</td>
            <td class="text-center font-mono">62.5%</td>
            <td class="text-center font-mono">63.0%</td>
            <td class="text-center font-mono">0.645</td>
            <td class="text-center font-mono">0.254</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">10.3</span>Asymmetric Dispatch Risk &amp; Residual Diagnostic Suite</h2>
      <div class="grid-2">
        <div>
          <div class="callout" style="margin-bottom: 4px;">
            <strong>Cost Asymmetry Analysis:</strong> In grid dispatch, False Negatives (predicting Down when demand Ramps Up) are 3x to 5x more costly than False Positives due to emergency peaker activation fines. CLM-Former minimized False Negatives to only 5 events (Recall: 84.4%), while Seasonal Naive suffered 12 critical under-generation events.
          </div>
          <p style="font-size: 7.4pt;">
            The 4-panel residual suite confirms that CLM-Former errors conform to an independent Gaussian distribution without unmodeled temporal correlation:
          </p>
          <ul style="font-size: 7.4pt; margin: 0; padding-left: 15px;">
            <li><strong>Ljung-Box Portmanteau Test:</strong> <i>Q</i> = 10.42, <i>p</i> = 0.9885 &gt; 0.05 &implies; Cannot reject null hypothesis of independent white noise.</li>
            <li><strong>Residual Normality:</strong> Q-Q plot exhibits <i>R</i><sup>2</sup> = 0.982 linearity; empirical mean is centered near zero (&minus;0.04M kW).</li>
          </ul>
        </div>
        <div class="fig-container" style="margin: 0;">
          <img src="{IMG6}" style="max-height: 230px;" alt="Figure 6: Residual Diagnostics Suite">
          <div class="fig-caption"><strong>Figure 6:</strong> 4-Panel CLM-Former Residual Diagnostic Suite.</div>
        </div>
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 11 of 12</span>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- PAGE 12: GRANULAR CLIENT CLUSTERING, CAUSALITY & VIVA STRATEGY             -->
<!-- ========================================================================= -->
<div class="page-last">
  <div class="page-content">
    <div>
      <div class="page-header">
        <span class="doc-title">Section 11: Granular Client Clustering, Explainability &amp; Academic Defense</span>
        <span>Review 1 &amp; Review 2 Consolidated Report</span>
      </div>

      <h2><span class="sec-badge">11.1</span>Disaggregated 370-Meter Customer Clustering (<i>K</i>-Means Archetypes)</h2>
      <p>
        To examine the micro-foundations of aggregate demand, we clustered all 370 client meters using <i>K</i>-Means (<i>k</i> = 4) based on scale (Mean Load), volatility (Coefficient of Variation CV = &sigma; / &mu;), and Weekend-to-Weekday consumption ratio:
      </p>

      <div class="fig-container">
        <img src="{IMG7}" style="max-height: 220px;" alt="Figure 7: Client Clustering & Archetypes">
        <div class="fig-caption"><strong>Figure 7:</strong> Disaggregated 370-Meter Customer Segmentation into 4 Archetypes &amp; Normalized 28-Day Representative Load Signatures.</div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Archetype Tier</th>
            <th>Customer Count</th>
            <th>Mean Daily Load (kW)</th>
            <th>Volatility (CV)</th>
            <th>Weekend Ratio</th>
            <th>Representative Behavioral Characteristics</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Tier 0: Industrial Baseload</strong></td>
            <td class="font-mono">42 (11.4%)</td>
            <td class="font-mono">48,210</td>
            <td class="font-mono">0.18</td>
            <td class="font-mono">0.96</td>
            <td>Continuous 24/7 manufacturing; flat predictable load; minimal weekend drop.</td>
          </tr>
          <tr>
            <td><strong>Tier 1: Commercial / SME</strong></td>
            <td class="font-mono">118 (31.9%)</td>
            <td class="font-mono">14,350</td>
            <td class="font-mono">0.42</td>
            <td class="font-mono">0.48</td>
            <td>Sharp 8 AM–6 PM daytime working peaks; massive 52% reduction on weekends.</td>
          </tr>
          <tr>
            <td><strong>Tier 2: Residential Small</strong></td>
            <td class="font-mono">164 (44.3%)</td>
            <td class="font-mono">3,120</td>
            <td class="font-mono">0.65</td>
            <td class="font-mono">1.12</td>
            <td>Dual morning (7–9 AM) and evening (6–10 PM) peaks; higher weekend consumption.</td>
          </tr>
          <tr>
            <td><strong>Tier 3: Intermittent Spikers</strong></td>
            <td class="font-mono">46 (12.4%)</td>
            <td class="font-mono">1,840</td>
            <td class="font-mono">1.34</td>
            <td class="font-mono">0.82</td>
            <td>High-volatility sporadic equipment; non-stationary stochastic spikes.</td>
          </tr>
        </tbody>
      </table>

      <h2><span class="sec-badge">11.2</span>Neural Granger Causality via Input-Output Jacobian Sensitivity</h2>
      <div class="grid-2">
        <div class="fig-container" style="margin: 0;">
          <img src="{IMG8}" style="max-height: 200px;" alt="Figure 8: Neural Granger Causality">
          <div class="fig-caption"><strong>Figure 8:</strong> Evaluation-Time Jacobian Sensitivity Matrix (|&part;<i>y&#770;</i> / &part;<i>x</i><sub><i>t</i>&minus;&tau;</sub>|).</div>
        </div>
        <div>
          <p style="font-size: 7.4pt; margin-top: 2px;">
            To mathematically verify feature attribution in black-box deep models, we computed the evaluation-time input-output Jacobian sensitivity gradient matrix:
            <div class="math-display"><i>J</i><sub>&tau;</sub> = (1 / <i>N</i><sub>test</sub>) &sum;<sub><i>i</i>=1</sub><sup><i>N</i><sub>test</sub></sup> |&part;<i>y&#770;</i><sup>(<i>i</i>)</sup> / &part;<i>x</i><sub><i>t</i>&minus;&tau;</sub><sup>(<i>i</i>)</sup>|, &emsp; &tau; &isin; {1, 2, &hellip;, 28}</div>
          </p>
          <div class="callout" style="font-size: 7.2pt;">
            <strong>Verification of Weekly Harmonic Resonance:</strong> As visualized in Figure 8, gradient magnitude peaks sharply at Lag 1 (persistence) and exhibits prominent harmonic secondary pulses at <strong>Lag 7, Lag 14, Lag 21, and Lag 28</strong>. This mathematically proves that CLM-Former's attention heads learned genuine multi-week calendar recurrence rather than fitting spurious noise.
          </div>
        </div>
      </div>

      <h2><span class="sec-badge">11.3</span>Academic Defense &amp; Viva Voce Strategy (High-Yield Technical Q&amp;A)</h2>
      <div class="grid-2" style="font-size: 7.2pt;">
        <div class="card" style="margin-bottom: 2px; padding: 4px 6px;">
          <strong>Q1: Why do ARIMA and Holt-Winters look like flat straight lines in multi-step forecasting?</strong><br>
          <em>Answer:</em> Future error shocks have zero expectation (E[&epsilon;] = 0), causing MA terms to vanish after <i>q</i> steps. Stationary AR terms decay exponentially (&phi;<sup><i>h</i></sup> &rarr; 0), leaving only constant drift &mu; &asymp; 0.
        </div>
        <div class="card" style="margin-bottom: 2px; padding: 4px 6px;">
          <strong>Q2: Why did XGBoost outperform Deep Learning at H=1, but lose at H=14?</strong><br>
          <em>Answer:</em> At <i>H</i> = 1, yesterday's load (<i>y</i><sub><i>t</i>&minus;1</sub>) is an immediate persistence oracle (<i>r</i> &gt; 0.85). However, at <i>H</i> = 14, recursive rollouts compound errors, whereas CLM-Former's multi-scale memory maintains stable accuracy.
        </div>
      </div>
    </div>

    <div class="page-footer">
      <span>Consolidated Technical Case Study Report | Smart Grid Load Forecasting</span>
      <span class="page-num">Page 12 of 12</span>
    </div>
  </div>
</div>

</body>
</html>
"""

# Replace image placeholders
html_content = html_template.replace('{IMG1}', img1) \
                            .replace('{IMG2}', img2) \
                            .replace('{IMG3}', img3) \
                            .replace('{IMG4}', img4) \
                            .replace('{IMG5}', img5) \
                            .replace('{IMG6}', img6) \
                            .replace('{IMG7}', img7) \
                            .replace('{IMG8}', img8)

# Save HTML
html_path = 'docs/Review1_and_Review2_Comprehensive_Report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"Generated {html_path} ({os.path.getsize(html_path) / 1024:.1f} KB)")

# Generate PDF via headless Edge
pdf_path = 'docs/Review1_and_Review2_Comprehensive_Report.pdf'
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.path.exists(edge_path):
    print("Generating PDF via headless Microsoft Edge...")
    cmd = [
        edge_path,
        '--headless',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        '--no-pdf-header-footer',
        f'--print-to-pdf={os.path.abspath(pdf_path)}',
        os.path.abspath(html_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Check page count
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    page_count = len(re.findall(rb'/Type\s*/Page\b', pdf_bytes))
    print(f"SUCCESS: Generated PDF report: {pdf_path} ({os.path.getsize(pdf_path) / 1024:.1f} KB) | EXACT PAGE COUNT: {page_count} PAGES!")
else:
    print("Edge not found at default location.")
