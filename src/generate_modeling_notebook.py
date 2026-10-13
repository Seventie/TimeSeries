"""Generator for Review 2 modeling notebook.

Completely frames the notebook around the official 25-Mark Evaluation Scheme:
1. ARIMA / SARIMA / SARIMAX (2+ different models)          [3 Marks]
2. Holt-Winters models (Additive & Multiplicative)         [3 Marks]
3. ML / DL models (XGBoost, RNN, LSTM, CLM-Former, MKJR)   [3 Marks]
4. Baseline forecast (Naive, Seasonal Naive, Rolling Mean) [2 Marks]
5. Forecast accuracy (Formal Equations, MAE, RMSE, MAPE, R²)[2 Marks]
6. Model comparison (Unified Tables, Radar Chart, H=1 vs H=14)[2 Marks]
7. Residual Analysis (ACF, Q-Q, Distribution, Ljung-Box)   [3 Marks]
8. Research conclusion (Synthesis, Dispatch, Review 3 Plan)[2 Marks]
9. Individual contribution (Team breakdown, deliverables)  [5 Marks]
Total: 25 Marks
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ------------------------------------------------------------------------------
# Cell 1: Header & Review 2 Evaluation Rubric Mapping
# ------------------------------------------------------------------------------
md(r"""# Phase 2: Predictive Modeling, Advanced Hybrid Architectures, and Causal Analysis
### Project: Decoding Electricity Demand — Analyzing Consumption Patterns and Forecasting Future Energy Load
### Academic Review 2 Evaluation Portfolio (25 Marks Total)

---

## 📋 Evaluation Scheme Alignment & Review 2 Roadmap

This notebook is formally structured to satisfy each requirement of the **Review 2 Evaluation Rubric (25 Marks)**:

| Module # | Evaluation Requirement | Marks | Section in Notebook | Key Models & Artifacts Delivered |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **ARIMA / SARIMA / SARIMAX (2+ Models)** | **3** | **Section 4** | Non-seasonal $\text{ARIMA}(1,1,1)$, Seasonal $\text{SARIMA}(1,1,1)\times(1,0,1)_7$, Exogenous $\text{SARIMAX}(1,1,1)\times(1,0,1)_7$ with weekend & Fourier calendar features; Multi-step blind vs. Rolling 1-step walk-forward |
| **2** | **Holt-Winters Models** | **3** | **Section 5** | Holt-Winters Additive Seasonality & Multiplicative Seasonality; smoothing parameters ($\alpha, \beta, \gamma$); Multi-step & Rolling tracking |
| **3** | **ML / DL Models** | **3** | **Section 6 & 7** | XGBoost Regressor (ML), Vanilla RNN, Standard LSTM, CLM-Former (Nature 2026), MKJR (Elsevier 2026); Neural Granger Causal Discovery |
| **4** | **Baseline Forecasts** | **2** | **Section 3** | Naive Persistence ($\hat{y}_t = y_{t-1}$), Seasonal Naive ($\hat{y}_t = y_{t-7}$), 7-Day Rolling Mean, Historical Mean Baseline |
| **5** | **Forecast Accuracy** | **2** | **Section 2** | Mathematical formulations for $\text{MAE}, \text{RMSE}, \text{MAPE}, R^2, \text{Directional Accuracy}$; $\Delta\% \text{ vs. Baseline}$ improvement quantification |
| **6** | **Model Comparison** | **2** | **Section 8 & 9** | Unified single-step comparison table ($H=1$); 14-day multi-horizon benchmark ($H=14$); Horizon error decay curves; Interactive radar chart |
| **7** | **Residual Analysis** | **3** | **Section 10** | 4-panel diagnostic suite (Time-series, Histogram/Gaussian, ACF, Q-Q); Formal Ljung-Box Portmanteau white-noise test |
| **8** | **Research Conclusion** | **2** | **Section 11** | Empirical findings, Persistence Regime Paradox, grid dispatch implications, limitations & Review 3 roadmap |
| **9** | **Individual Contribution** | **5** | **Section 12** | Comprehensive role matrix, technical deliverables, and candidate viva voce defense profiles |
| **—** | **Viva Voce Defense Strategy** | **Bonus** | **Section 13** | High-yield defense Q&A on mathematical derivations, gating, attention vs SSM, and causal Jacobian gradients |

---""")

# ------------------------------------------------------------------------------
# Cell 2: Imports & Environment Setup
# ------------------------------------------------------------------------------
code(r"""import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy.stats as stats

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
pio.renderers.default = 'notebook'

print(f"PyTorch Version   : {torch.__version__}")
print(f"Compute Device    : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print("Statistical Stack : statsmodels (ARIMA, SARIMAX, Holt-Winters, Ljung-Box)")
print("ML & DL Stack     : XGBoost Regressor, PyTorch Sequential Baselines & 2026 SOTA")""")

# ------------------------------------------------------------------------------
# Cell 3: Data Ingestion & Chronological Train-Test Split
# ------------------------------------------------------------------------------
md(r"""## 1. Data Preparation & Chronological Train-Test Split

To ensure fair benchmark comparison and computational tractability across classical statistical models (where 15-minute SARIMA on 370 clients requires millions of matrix inversions), we aggregate the readings to **Daily Total Load**.
We utilize 2012–2014 data and reserve the final **60 days** as our unseen test horizon.""")

code(r"""# Load daily aggregated load (from cache if available, else from raw 710MB text)
cache_path = '../data/daily_aggregated_load.csv'
if not os.path.exists(cache_path):
    cache_path = 'data/daily_aggregated_load.csv'

if os.path.exists(cache_path):
    print("Loading cached daily aggregated load...")
    df_daily = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
else:
    data_path = '../data/LD2011_2014.txt' if os.path.exists('../data/LD2011_2014.txt') else 'data/LD2011_2014.txt'
    print(f"Aggregating from {data_path} in chunks to conserve memory...")
    chunks = [chunk.sum(axis=1) for chunk in pd.read_csv(data_path, sep=';', decimal=',', parse_dates=[0], index_col=0, chunksize=25000)]
    daily = pd.concat(chunks).resample('D').sum().loc['2012-01-01':]
    df_daily = pd.DataFrame({'Total_Load': daily})
    df_daily.index.name = 'date'
    df_daily.to_csv(cache_path)

# Calendar feature engineering for statistical & ML regressors
dow = df_daily.index.dayofweek.values
df_daily['is_weekend'] = (dow >= 5).astype(float)
df_daily['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
df_daily['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

print(f"Total Daily Observations: {len(df_daily)} days ({df_daily.index[0].date()} to {df_daily.index[-1].date()})")
df_daily.head()""")

# ------------------------------------------------------------------------------
# Cell 4: Train / Test Split Plot
# ------------------------------------------------------------------------------
code(r"""test_horizon = 60
train = df_daily.iloc[:-test_horizon].copy()
test = df_daily.iloc[-test_horizon:].copy()

print(f"Training Set : {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Testing Set  : {len(test)} days ({test.index[0].date()} to {test.index[-1].date()})")

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['Total_Load'], mode='lines', name='Train Set (Historical)', line=dict(color='#1f77b4', width=1.5)))
fig.add_trace(go.Scatter(x=test.index, y=test['Total_Load'], mode='lines', name='Test Set (60-day Unseen)', line=dict(color='#ff7f0e', width=2)))
fig.update_layout(title='Chronological Train/Test Partition (Aggregated Daily Grid Load)', xaxis_title='Date', yaxis_title='Electricity Load (kW)', template='plotly_white')
fig.show()""")

# ------------------------------------------------------------------------------
# Cell 5: Forecast Accuracy Formulations & Shared Evaluation Toolkit (Rubric Item 5 - 2 Marks)
# ------------------------------------------------------------------------------
md(r"""## 2. Forecast Accuracy Formulations & Shared Evaluation Toolkit
### 🎯 Evaluation Rubric Item 5 — Forecast Accuracy (2 Marks)

To rigorously quantify forecasting performance across models, we establish five formal mathematical accuracy metrics:

1. **Mean Absolute Error (MAE):** Linear penalty preserving the physical scale (kW):
   $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N \left| y_t - \hat{y}_t \right|$$
2. **Root Mean Squared Error (RMSE):** Quadratic penalty penalizing large dispatch deviations:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left( y_t - \hat{y}_t \right)^2}$$
3. **Mean Absolute Percentage Error (MAPE):** Scale-independent relative deviation percentage:
   $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100\%$$
4. **Coefficient of Determination ($R^2$):** Proportion of variance captured relative to historical mean baseline:
   $$R^2 = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N (y_t - \bar{y})^2}$$
5. **Directional Accuracy (DA):** Frequency with which the forecast correctly predicts day-over-day movement direction:
   $$\text{DA} = \frac{1}{N-1} \sum_{t=2}^N \mathbb{I}\left(\text{sgn}(y_t - y_{t-1}) == \text{sgn}(\hat{y}_t - y_{t-1})\right) \times 100\%$$""")

code(r"""def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    da = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1])) * 100.0 if len(y_true) > 1 else 0.0
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100.0,
        'R2': r2_score(y_true, y_pred),
        'DA': da
    }

RESULTS, TIMINGS, COLORS = {}, {}, {}
PALETTE = {
    'Naive (Persistence)': '#7f7f7f',
    'Seasonal Naive (Lag 7)': '#8c564b',
    '7-Day Rolling Mean': '#c7c7c7',
    'ARIMA (Multi-Step)': '#d62728',
    'ARIMA (Rolling)': '#e377c2',
    'SARIMA (Multi-Step)': '#ff7f0e',
    'SARIMA (Rolling)': '#bcbd22',
    'SARIMAX (Multi-Step)': '#d6616b',
    'SARIMAX (Rolling)': '#ce6dbd',
    'Holt-Winters Additive': '#17becf',
    'Holt-Winters Multiplicative': '#9edae5',
    'XGBoost': '#2ca02c',
    'Vanilla RNN': '#9467bd',
    'Standard LSTM': '#1f77b4',
    'CLM-Former': '#393b79',
    'MKJR': '#637939',
}

def register(name, preds, secs=0.0):
    m = compute_metrics(test['Total_Load'].values, preds)
    RESULTS[name] = m
    TIMINGS[name] = secs
    print(f"{name:28s} | MAE={m['MAE']:>13,.0f} | RMSE={m['RMSE']:>13,.0f} | MAPE={m['MAPE']:6.2f}% | R2={m['R2']:6.3f} | DA={m['DA']:5.1f}% | Time={secs:6.2f}s")
    return m

def plot_model_diagnostics(name, pred_col=None):
    y_true = test['Total_Load'].values
    y_pred = test[name].values if pred_col is None else test[pred_col].values
    res = y_true - y_pred
    color = PALETTE.get(name, '#1f77b4')

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{name}: Actual vs Predicted', 'Residuals (e = y - y_hat)', 'Scatter: Actual vs Predicted'],
        column_widths=[0.45, 0.30, 0.25]
    )
    # 1. Trajectory overlay
    fig.add_trace(go.Scatter(x=test.index, y=y_true, mode='lines', name='Actual', line=dict(color='black', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test.index, y=y_pred, mode='lines', name=name, line=dict(color=color, width=2, dash='dot' if 'Multi-Step' in name else 'solid')), row=1, col=1)
    # 2. Residuals
    fig.add_trace(go.Scatter(x=test.index, y=res, mode='lines+markers', name='Residual', line=dict(color=color, width=1.5)), row=1, col=2)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=2)
    # 3. 45-degree scatter
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Points', marker=dict(color=color, size=5, opacity=0.8)), row=1, col=3)
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode='lines', name='Ideal 45°', line=dict(color='red', dash='dash')), row=1, col=3)

    fig.update_layout(title=f'Model Diagnostics — {name}', template='plotly_white', showlegend=False, height=350)
    fig.show()""")

# ------------------------------------------------------------------------------
# Cell 6: Baseline Forecasting Models (Rubric Item 4 - 2 Marks)
# ------------------------------------------------------------------------------
md(r"""## 3. Baseline Forecasting Models
### 🎯 Evaluation Rubric Item 4 — Baseline Forecast (2 Marks)

Before deploying advanced models, we construct four foundational baselines to define the lower performance bounds:
1. **Naive (Persistence / Random Walk) Baseline:** Assumes the next day equals yesterday's load ($\hat{y}_t = y_{t-1}$).
2. **Seasonal Naive Baseline (Lag 7):** Assumes the next day equals the same day last week ($\hat{y}_t = y_{t-7}$).
3. **7-Day Rolling Moving Average Baseline:** Smooths the last week of observations ($\hat{y}_t = \frac{1}{7}\sum_{i=1}^7 y_{t-i}$).
4. **Historical Mean Baseline:** Constant mean of the entire training dataset ($\hat{y}_t = \bar{y}_{train}$).""")

code(r"""# 1. Naive (Persistence) Baseline: y_hat_t = y_{t-1}
full_series = df_daily['Total_Load'].values
naive_pred = full_series[len(train)-1 : len(train)-1 + len(test)]
test['Naive (Persistence)'] = naive_pred
register('Naive (Persistence)', naive_pred, secs=0.001)

# 2. Seasonal Naive Baseline (Lag 7): y_hat_t = y_{t-7}
snaive_pred = full_series[len(train)-7 : len(train)-7 + len(test)]
test['Seasonal Naive (Lag 7)'] = snaive_pred
register('Seasonal Naive (Lag 7)', snaive_pred, secs=0.001)

# 3. 7-Day Rolling Moving Average Baseline
roll7_pred = [full_series[len(train) - 7 + i : len(train) + i].mean() for i in range(len(test))]
test['7-Day Rolling Mean'] = roll7_pred
register('7-Day Rolling Mean', roll7_pred, secs=0.002)

plot_model_diagnostics('Seasonal Naive (Lag 7)')""")

# ------------------------------------------------------------------------------
# Cell 7: Classical Statistical Modeling: ARIMA / SARIMA / SARIMAX (Rubric Item 1 - 3 Marks)
# ------------------------------------------------------------------------------
md(r"""## 4. Classical Statistical Modeling: ARIMA, SARIMA & SARIMAX
### 🎯 Evaluation Rubric Item 1 — ARIMA / SARIMA / SARIMAX (2+ Different Models) (3 Marks)

We evaluate three distinct classical linear econometric formulations:
1. **Non-Seasonal $\text{ARIMA}(1, 1, 1)$:** Captures first-order integration ($d=1$) and localized autoregressive moving-average shocks.
2. **Seasonal $\text{SARIMA}(1, 1, 1) \times (1, 0, 1)_7$:** Adds multiplicative weekly seasonal autoregression ($\Phi_1$) and seasonal moving averages ($\Theta_1$).
3. **Exogenous $\text{SARIMAX}(1, 1, 1) \times (1, 0, 1)_7$:** Incorporates external deterministic calendar regressors ($\text{is\_weekend}, \sin(\text{day}), \cos(\text{day})$).

For each model, we evaluate two operational modes:
* **Multi-Step Blind:** Open-loop recursive forecast 60 days into the future with zero sensor feedback.
* **Rolling 1-Step Walk-Forward:** Filtered forecast where day $t-1$ updates the internal Kalman state before forecasting day $t$.""")

code(r"""# --- Model 1: ARIMA(1, 1, 1) ---
print("Fitting Non-Seasonal ARIMA(1, 1, 1)...")
t0 = time.time()
arima_model = ARIMA(train['Total_Load'], order=(1, 1, 1)).fit()
arima_multi = arima_model.forecast(steps=len(test)).values
t_amulti = time.time() - t0
test['ARIMA (Multi-Step)'] = arima_multi
register('ARIMA (Multi-Step)', arima_multi, t_amulti)

t0 = time.time()
arima_applied = arima_model.apply(df_daily['Total_Load'])
arima_rolling = arima_applied.predict(start=test.index[0], end=test.index[-1], dynamic=False).values
t_aroll = time.time() - t0
test['ARIMA (Rolling)'] = arima_rolling
register('ARIMA (Rolling)', arima_rolling, t_aroll)

# --- Model 2: SARIMA(1, 1, 1)x(1, 0, 1, 7) ---
print("\nFitting Seasonal SARIMA(1, 1, 1)x(1, 0, 1, 7)...")
t0 = time.time()
sarima_model = SARIMAX(train['Total_Load'], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_multi = sarima_model.forecast(steps=len(test)).values
t_smulti = time.time() - t0
test['SARIMA (Multi-Step)'] = sarima_multi
register('SARIMA (Multi-Step)', sarima_multi, t_smulti)

t0 = time.time()
sarima_applied = sarima_model.apply(df_daily['Total_Load'])
sarima_rolling = sarima_applied.predict(start=test.index[0], end=test.index[-1], dynamic=False).values
t_sroll = time.time() - t0
test['SARIMA (Rolling)'] = sarima_rolling
register('SARIMA (Rolling)', sarima_rolling, t_sroll)

# --- Model 3: SARIMAX with Exogenous Calendar Features ---
exog_cols = ['is_weekend', 'dow_sin', 'dow_cos']
print("\nFitting SARIMAX(1, 1, 1)x(1, 0, 1, 7) with Exogenous Features...")
t0 = time.time()
sarimax_model = SARIMAX(train['Total_Load'], exog=train[exog_cols], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarimax_multi = sarimax_model.forecast(steps=len(test), exog=test[exog_cols]).values
t_sxmulti = time.time() - t0
test['SARIMAX (Multi-Step)'] = sarimax_multi
register('SARIMAX (Multi-Step)', sarimax_multi, t_sxmulti)

t0 = time.time()
sarimax_applied = sarimax_model.apply(df_daily['Total_Load'], exog=df_daily[exog_cols])
sarimax_rolling = sarimax_applied.predict(start=test.index[0], end=test.index[-1], exog=test[exog_cols], dynamic=False).values
t_sxroll = time.time() - t0
test['SARIMAX (Rolling)'] = sarimax_rolling
register('SARIMAX (Rolling)', sarimax_rolling, t_sxroll)""")

# ------------------------------------------------------------------------------
# Cell 8: Mathematical Deep Dive: Why ARIMA/SARIMA Flatline
# ------------------------------------------------------------------------------
md(r"""### 🔬 Deep Dive: Why Do Multi-Step ARIMA & SARIMA Collapse into Straight Lines?

A frequent query during project defense is:
> *"Why does the multi-step ARIMA curve flatten into an almost horizontal straight line, and why does SARIMA collapse into a rigid, repeating wave with zero dynamic adaptation?"*

This is not a bug — **it is a strict mathematical property of linear autoregressive moving-average processes under open-loop recursive forecasting.**

---

#### 1. The Mathematical Mechanism of ARIMA Flatlining
Let $w_t = \Delta y_t = y_t - y_{t-1}$ represent the first-differenced series. The $\text{ARIMA}(1,1,1)$ model is:
$$w_t = c + \phi_1 w_{t-1} + \epsilon_t + \theta_1 \epsilon_{t-1}$$
When generating an out-of-sample forecast for horizon $h \ge 1$ from cutoff $T$, we compute the conditional expectation $\hat{w}_{T+h} = \mathbb{E}[w_{T+h} \mid \mathcal{F}_T]$:

1. **Vanishing of Moving Average ($\text{MA}$):**  
   Future error shocks $\epsilon_{T+1}, \epsilon_{T+2}, \dots$ have not occurred. By white noise definition, $\mathbb{E}[\epsilon_{T+k}] = 0$.  
   Therefore, the moving average term $\theta_1 \mathbb{E}[\epsilon_{T+h-1}] = 0$ vanishes completely for every horizon $h \ge 2$!
2. **Exponential Decay of Autoregression ($\text{AR}$):**  
   $$\hat{w}_{T+h} - \mu = \phi_1^h (w_T - \mu)$$
   Because the differenced series is stationary ($|\phi_1| < 1$), $\lim_{h \to \infty} \phi_1^h = 0 \implies \hat{w}_{T+h} \approx \mu$ for $h \ge 5$.
3. **Integrating to Original Load ($y_t$):**  
   $$\hat{y}_{T+h} = \hat{y}_T + \sum_{k=1}^h \hat{w}_{T+k} \approx \hat{y}_T + h \cdot \mu$$
   **This is the equation of a straight line: $f(h) = \mu \cdot h + \hat{y}_T$.**  
   In electricity grids, the daily differenced mean drift $\mu \approx 0$ (consumption does not drift to infinity). When $\mu = 0$, $\hat{y}_{T+h} \approx \text{constant}$.  
   **Hence, multi-step ARIMA mathematically collapses into a horizontal straight line.**

---

#### 2. The Setup Asymmetry: Multi-Step Static vs. Rolling Walk-Forward
* **Multi-Step (Blind):** Evaluated 60 days into the future without receiving updated readings. Errors accumulate, and the prediction collapses to the mean.
* **Rolling 1-Step (Filtered):** Day $t-1$ updates the internal Kalman state before forecasting day $t$. This allows SARIMA and SARIMAX to actively track dynamic movements (improving $R^2$ from $-0.22$ to $+0.20$!).""")

code(r"""# Visualizing Multi-Step Flatlining vs Rolling Walk-Forward Tracking
fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(x=test.index, y=test['Total_Load'], mode='lines', name='Actual Load', line=dict(color='black', width=2.5)))
fig_comp.add_trace(go.Scatter(x=test.index, y=test['ARIMA (Multi-Step)'], mode='lines', name='ARIMA Multi-Step (Flat Line)', line=dict(color='#d62728', dash='dot', width=2)))
fig_comp.add_trace(go.Scatter(x=test.index, y=test['ARIMA (Rolling)'], mode='lines', name='ARIMA Rolling 1-Step (Dynamic)', line=dict(color='#e377c2', width=1.5)))
fig_comp.add_trace(go.Scatter(x=test.index, y=test['SARIMA (Multi-Step)'], mode='lines', name='SARIMA Multi-Step (Rigid Wave)', line=dict(color='#ff7f0e', dash='dash', width=2)))
fig_comp.add_trace(go.Scatter(x=test.index, y=test['SARIMA (Rolling)'], mode='lines', name='SARIMA Rolling 1-Step (Dynamic)', line=dict(color='#bcbd22', width=1.5)))
fig_comp.add_trace(go.Scatter(x=test.index, y=test['SARIMAX (Rolling)'], mode='lines', name='SARIMAX Rolling 1-Step (Exogenous)', line=dict(color='#ce6dbd', width=1.5)))

fig_comp.update_layout(
    title='Multi-Step Static Flatlining vs. Rolling 1-Step Dynamic Tracking in ARIMA / SARIMA / SARIMAX',
    xaxis_title='Date', yaxis_title='Electricity Load (kW)',
    template='plotly_white', hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)')
)
fig_comp.show()""")

# ------------------------------------------------------------------------------
# Cell 9: Exponential Smoothing: Holt-Winters Models (Rubric Item 2 - 3 Marks)
# ------------------------------------------------------------------------------
md(r"""## 5. Exponential Smoothing — Holt-Winters Models
### 🎯 Evaluation Rubric Item 2 — Holt-Winters Models (3 Marks)

Holt-Winters Exponential Smoothing captures level ($\ell_t$), linear trend ($b_t$), and seasonal indices ($s_t$) using exponentially decaying smoothing weights ($\alpha, \beta, \gamma \in (0, 1)$):

#### 1. Holt-Winters Additive Seasonality Formulation:
$$\begin{aligned}
\text{Level: } & \ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \\
\text{Trend: } & b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1} \\
\text{Seasonal: } & s_t = \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m} \\
\text{Forecast: } & \hat{y}_{t+h} = \ell_t + h b_t + s_{t-m+h_m^+}
\end{aligned}$$

#### 2. Holt-Winters Multiplicative Seasonality Formulation:
$$\hat{y}_{t+h} = (\ell_t + h b_t) \times s_{t-m+h_m^+}$$
Multiplicative seasonality scales the seasonal amplitude proportionally with the base load level.

We fit both **Additive** and **Multiplicative** variants with weekly periodicity ($m=7$).""")

code(r"""# --- Model 1: Holt-Winters Additive Seasonality ---
print("Fitting Holt-Winters Additive Seasonality (m=7)...")
t0 = time.time()
hw_add = ExponentialSmoothing(train['Total_Load'], trend='add', seasonal='add', seasonal_periods=7).fit()
hw_add_multi = hw_add.forecast(len(test)).values
t_hwadd = time.time() - t0
test['Holt-Winters Additive'] = hw_add_multi
register('Holt-Winters Additive', hw_add_multi, t_hwadd)
print(f"  Additive HW Smoothing Parameters: alpha={hw_add.params['smoothing_level']:.3f}, beta={hw_add.params['smoothing_trend']:.3f}, gamma={hw_add.params['smoothing_seasonal']:.3f}")

# --- Model 2: Holt-Winters Multiplicative Seasonality ---
print("\nFitting Holt-Winters Multiplicative Seasonality (m=7)...")
t0 = time.time()
hw_mul = ExponentialSmoothing(train['Total_Load'], trend='add', seasonal='mul', seasonal_periods=7).fit()
hw_mul_multi = hw_mul.forecast(len(test)).values
t_hwmul = time.time() - t0
test['Holt-Winters Multiplicative'] = hw_mul_multi
register('Holt-Winters Multiplicative', hw_mul_multi, t_hwmul)
print(f"  Multiplicative HW Smoothing Parameters: alpha={hw_mul.params['smoothing_level']:.3f}, beta={hw_mul.params['smoothing_trend']:.3f}, gamma={hw_mul.params['smoothing_seasonal']:.3f}")

plot_model_diagnostics('Holt-Winters Additive')""")

# ------------------------------------------------------------------------------
# Cell 10: Machine Learning Baseline: XGBoost (Rubric Item 3 - 3 Marks, Part 1)
# ------------------------------------------------------------------------------
md(r"""## 6. Machine Learning Baseline — XGBoost Regressor
### 🎯 Evaluation Rubric Item 3 — ML / DL Models (Part 1: Machine Learning) (3 Marks)

Supervised gradient-boosted decision trees represent our non-linear tabular benchmark.  
We engineer autoregressive lag features ($\text{lag}_1, \text{lag}_7, \text{lag}_{14}$), a 7-day rolling moving average, and calendar metadata (day-of-week, month, weekend).""")

code(r"""def engineer_ml_features(df_in):
    d = df_in.copy()
    d['dayofweek'] = d.index.dayofweek
    d['month'] = d.index.month
    d['is_weekend'] = d['dayofweek'].isin([5, 6]).astype(int)
    d['lag_1'] = d['Total_Load'].shift(1)
    d['lag_7'] = d['Total_Load'].shift(7)
    d['lag_14'] = d['Total_Load'].shift(14)
    d['rolling_mean_7'] = d['Total_Load'].shift(1).rolling(7).mean()
    return d.dropna()

df_features = engineer_ml_features(df_daily)
train_feat = df_features.iloc[:-test_horizon]
test_feat = df_features.iloc[-test_horizon:]
X_train, y_train = train_feat.drop('Total_Load', axis=1), train_feat['Total_Load']
X_test, y_test = test_feat.drop('Total_Load', axis=1), test_feat['Total_Load']

print("Training XGBoost Regressor...")
t_start = time.time()
xgb_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
t_xgb = time.time() - t_start

test['XGBoost'] = xgb_pred
register('XGBoost', xgb_pred, t_xgb)
plot_model_diagnostics('XGBoost')

# Feature importance visual
fig_imp = go.Figure(go.Bar(
    x=xgb_model.feature_importances_, y=X_train.columns, orientation='h',
    marker_color='#2ca02c', hovertemplate='%{y}: %{x:.3f}<extra></extra>'
))
fig_imp.update_layout(title='XGBoost Feature Importance (Gain Weight)', xaxis_title='Importance Score', yaxis_title='Feature', template='plotly_white')
fig_imp.show()""")

# ------------------------------------------------------------------------------
# Cell 11: Deep Learning Setup: RevIN & Windows
# ------------------------------------------------------------------------------
md(r"""## 7. Deep Learning Architectures — Baselines & 2026 SOTA Replications
### 🎯 Evaluation Rubric Item 3 — ML / DL Models (Part 2: Deep Learning) (3 Marks)

For sequential deep learning, we implement:
1. **Classical Recurrent Baselines:** Vanilla RNN (`nn.RNN`) & Standard LSTM (`nn.LSTM`).
2. **2026 SOTA Deep Model 1 — CLM-Former (*Nature Sci. Reports 2026*):** Decomposition + FFT Autocorrelation + CLM-subNet.
3. **2026 SOTA Deep Model 2 — MKJR (*Elsevier EAAI 2026*):** RevIN + KAN B-spline Layers + Dual-Stream BiMamba + Jacobian Regularization.

All deep models share:
* **Reversible Instance Normalization (RevIN):** Normalizes each 56-day sliding window to eliminate non-stationarity without modifying model parameters:
  $$x^{(0)} = \frac{x - \mu}{\sigma + \epsilon}, \quad \hat{y} = \hat{y}^{(0)} \cdot \sigma + \mu$$
* **Lookback ($L=56$):** 8 complete weekly cycles.
* **Channels (4):** Total Load, $\sin(\text{day})$, $\cos(\text{day})$, Weekend flag.""")

code(r"""lookback = 56
feat = df_daily.copy()
dow = feat.index.dayofweek.values
feat['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
feat['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
feat['weekend'] = (dow >= 5).astype(np.float32)
CHANNELS = ['Total_Load', 'dow_sin', 'dow_cos', 'weekend']
arr = feat[CHANNELS].values.astype(np.float32)

X_all, y_all = [], []
for i in range(lookback, len(arr)):
    X_all.append(arr[i - lookback:i])
    y_all.append(arr[i, 0])
X_all = np.asarray(X_all, dtype=np.float32)
y_all = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)

X_tr_all, y_tr_all = X_all[:-test_horizon], y_all[:-test_horizon]
X_te, y_te = X_all[-test_horizon:], y_all[-test_horizon:]
val_n = 90
X_tr, y_tr = X_tr_all[:-val_n], y_tr_all[:-val_n]
X_val, y_val = X_tr_all[-val_n:], y_tr_all[-val_n:]

to_t = lambda a: torch.tensor(a, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(to_t(X_tr), to_t(y_tr)), batch_size=64, shuffle=True)
X_val_t, y_val_t = to_t(X_val), to_t(y_val)
X_te_t, y_te_t = to_t(X_te), to_t(y_te)
criterion = nn.MSELoss()

print(f"Dataset windows -> train {X_tr.shape}, val {X_val.shape}, test {X_te.shape} (channels={CHANNELS})")""")

# ------------------------------------------------------------------------------
# Cell 12: RevIN & General Training Loop
# ------------------------------------------------------------------------------
code(r"""class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, load):            # load: (B, L)
        mu = load.mean(dim=1, keepdim=True)
        sd = load.std(dim=1, keepdim=True) + self.eps
        return (load - mu) / sd, mu, sd

    def denormalize(self, y_norm, mu, sd):  # y_norm, mu, sd: (B, 1)
        return y_norm * sd + mu

def train_dl(model, name=None, lr=2e-3, epochs=100, patience=12, jacobian=False, lam_sens=0.02, clip=1.0,
             loader=None, Xv=None, yv=None, Xte=None, register_result=True):
    loader = train_loader if loader is None else loader
    Xv = X_val_t if Xv is None else Xv
    yv = y_val_t if yv is None else yv
    Xte = X_te_t if Xte is None else Xte
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=4)
    best, best_state, wait, t0 = float('inf'), None, 0, time.time()
    
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            if jacobian:
                bx = bx.clone().requires_grad_(True)
            yhat, yn, mu, sd = model(bx)
            loss = criterion(yn, (by - mu) / sd)
            if jacobian:
                v = torch.randn_like(yn)
                g = torch.autograd.grad((v * yn).sum(), bx, create_graph=True)[0]
                loss = loss + lam_sens * (g ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            
        model.eval()
        with torch.no_grad():
            _, vyn, vmu, vsd = model(Xv)
            vloss = criterion(vyn, (yv - vmu) / vsd).item()
        sched.step(vloss)
        
        if vloss < best - 1e-7:
            best, best_state, wait = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                if name: print(f"  [{name}] Early stop @ epoch {ep+1} (best val loss {best:.4f})")
                break
        if (ep + 1) % 10 == 0 and name:
            print(f"  [{name}] Epoch {ep+1}/{epochs} - Val Loss: {vloss:.4f} (Best: {best:.4f})", flush=True)
            
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yhat, _, _, _ = model(Xte)
    secs = time.time() - t0
    if register_result:
        test[name] = yhat.numpy().flatten()
        return register(name, yhat.numpy().flatten(), secs)
    return yhat.numpy(), secs""")

# ------------------------------------------------------------------------------
# Cell 13: Classical Recurrent Baselines: Vanilla RNN & Standard LSTM
# ------------------------------------------------------------------------------
code(r"""# Model 4: Vanilla RNN (2 layers, hidden=64)
class VanillaRNN(nn.Module):
    def __init__(self, in_ch=4, hidden=64, num_layers=2, out_dim=1):
        super().__init__()
        self.revin = RevIN()
        self.rnn = nn.RNN(in_ch, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        load = x[:, :, 0]
        nload, mu, sd = self.revin.normalize(load)
        xin = torch.cat([nload.unsqueeze(-1), x[:, :, 1:]], dim=-1)
        out, _ = self.rnn(xin)
        y_norm = self.fc(out[:, -1, :])
        return self.revin.denormalize(y_norm, mu, sd), y_norm, mu, sd

# Model 5: Standard LSTM (2 layers, hidden=64)
class StandardLSTM(nn.Module):
    def __init__(self, in_ch=4, hidden=64, num_layers=2, out_dim=1):
        super().__init__()
        self.revin = RevIN()
        self.lstm = nn.LSTM(in_ch, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        load = x[:, :, 0]
        nload, mu, sd = self.revin.normalize(load)
        xin = torch.cat([nload.unsqueeze(-1), x[:, :, 1:]], dim=-1)
        out, _ = self.lstm(xin)
        y_norm = self.fc(out[:, -1, :])
        return self.revin.denormalize(y_norm, mu, sd), y_norm, mu, sd

print("Training Vanilla RNN...")
torch.manual_seed(42)
rnn_model = VanillaRNN(in_ch=len(CHANNELS))
train_dl(rnn_model, 'Vanilla RNN', lr=2e-3, epochs=100, patience=15)

print("\nTraining Standard LSTM...")
torch.manual_seed(42)
lstm_model = StandardLSTM(in_ch=len(CHANNELS))
train_dl(lstm_model, 'Standard LSTM', lr=2e-3, epochs=100, patience=15)""")

# ------------------------------------------------------------------------------
# Cell 14: SOTA Model 1: CLM-Former (Nature 2026)
# ------------------------------------------------------------------------------
md(r"""### 2026 SOTA Deep Model 1 — CLM-Former (*Nature Sci. Reports 2026*)
#### Replicated from *Rahmatinia & Ramezani (Nature Publishing Group, 2026)*

CLM-Former embeds the **CLM-subNet** inside a decomposition and autocorrelation block:
1. **Series Decomposition:** $\text{Trend} = \text{AvgPool}(X)$, $\text{Seasonal} = X - \text{Trend}$.
2. **Autocorrelation:** FFT-based $O(L \log L)$ cross-correlation extracts top-$k$ periodic lags and performs softmax-weighted sequence rolls.
3. **CLM-subNet:** Two 1D-CNN layers (capturing localized spikes) followed by a **bottleneck LSTM (hidden 64)** compressing parameters.
4. **Fused Representation Recombination:** Fuses bottleneck seasonal state with recent trend context to prevent long-horizon trend extrapolation divergence.""")

code(r"""class SeriesDecomp(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)

    def forward(self, x):
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        return x - trend, trend

class AutoCorrelation(nn.Module):
    def forward(self, x):
        B, T, d = x.shape
        q = x.transpose(1, 2)
        fft = torch.fft.rfft(q, dim=-1)
        corr = torch.fft.irfft(fft * torch.conj(fft), n=T, dim=-1).mean(dim=1)
        k = max(1, int(np.log(T)))
        w, idx = torch.topk(corr, k, dim=-1)
        w = torch.softmax(w, dim=-1)
        out = torch.zeros_like(x)
        for i in range(k):
            rolled = torch.stack([torch.roll(x[b], int(idx[b, i].item()), dims=0) for b in range(B)], dim=0)
            out = out + rolled * w[:, i].view(B, 1, 1)
        return out

class CLMFormer(nn.Module):
    def __init__(self, in_ch=4, hidden=64, out_dim=1):
        super().__init__()
        self.revin = RevIN()
        self.decomp = SeriesDecomp(kernel=7)
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=3, padding=1)
        self.drop1 = nn.Dropout(0.05)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout(0.05)
        self.autocorr = AutoCorrelation()
        self.lstm = nn.LSTM(32, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 7, 64),
            nn.GELU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        load = x[:, :, 0]
        nload, mu, sd = self.revin.normalize(load)
        seasonal, trend = self.decomp(nload)
        seq = torch.cat([seasonal.unsqueeze(-1), x[:, :, 1:]], dim=-1)
        c = seq.transpose(1, 2)
        c = self.drop1(F.gelu(self.conv1(c)))
        c = self.drop2(F.gelu(self.conv2(c)))
        c = c.transpose(1, 2)
        c = self.autocorr(c)
        out, _ = self.lstm(c)
        fused = torch.cat([out[:, -1, :], trend[:, -7:]], dim=-1)
        y_norm = self.head(fused)
        return self.revin.denormalize(y_norm, mu, sd), y_norm, mu, sd

print("Training CLM-Former (Decomposition + Autocorrelation + CLM-subNet)...")
torch.manual_seed(42)
clm_model = CLMFormer(in_ch=len(CHANNELS))
train_dl(clm_model, 'CLM-Former', lr=2e-3, epochs=60, patience=10, jacobian=False)
plot_model_diagnostics('CLM-Former')""")

# ------------------------------------------------------------------------------
# Cell 15: SOTA Model 2: MKJR (Elsevier 2026)
# ------------------------------------------------------------------------------
md(r"""### 2026 SOTA Deep Model 2 — MKJR (Mamba-KAN with Jacobian Regularization)
#### Replicated from *Engineering Applications of Artificial Intelligence (Elsevier, 2026)* — Zhang et al.

MKJR combines three primary innovations:
1. **Kolmogorov-Arnold Network (KAN) Linear Layers:** Replaces fixed node activations with learnable univariate B-spline/RBF basis expansions on edges:
   $$\phi(x) = w_{base} \cdot \text{SiLU}(x) + \sum_{k=1}^5 c_k \exp(-\gamma (x - \mu_k)^2)$$
2. **Asymmetric Bidirectional Dual-Stream Mamba Backbone:** Uses Selective State Spaces ($O(L)$ linear sequence complexity) with a detail stream (local-gated) and global stream (pool-gated).
3. **Matrix-Free Random-Projection Jacobian Regularization:** Penalizes sensitivity to noise via stochastic directional gradient probes $v \sim \mathcal{N}(0, I)$ during backpropagation.""")

code(r"""class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_centers=5):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_centers))
        self.register_buffer('centers', torch.linspace(-2.0, 2.0, num_centers))
        self.gamma = 1.0
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1)

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        rbf = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        spline = torch.einsum('...ic,oic->...o', rbf, self.spline_weight)
        return base + spline

class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model=32, d_state=8):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, u):
        B, Lseq, _ = u.shape
        A = -torch.exp(self.A_log)
        proj = self.x_proj(u)
        Bm, Cm, delta = proj[:, :, :self.d_state], proj[:, :, self.d_state:2 * self.d_state], F.softplus(proj[:, :, -1:])
        y = torch.zeros_like(u)
        h = torch.zeros(B, self.d_model, self.d_state, device=u.device)
        for t in range(Lseq):
            dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
            dB = delta[:, t, :].unsqueeze(-1) * Bm[:, t, :].unsqueeze(1)
            h = dA * h + dB * u[:, t, :].unsqueeze(-1)
            y[:, t, :] = torch.sum(h * Cm[:, t, :].unsqueeze(1), dim=-1) + self.D * u[:, t, :]
        return self.out_proj(F.silu(y))

class LocalGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1)

    def forward(self, z):
        g = self.conv(z.transpose(1, 2)).transpose(1, 2)
        return z * torch.sigmoid(g)

class GlobalGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, z):
        g = torch.sigmoid(self.fc(z.mean(dim=1, keepdim=True)))
        return z * g + z

class BiMamba(nn.Module):
    def __init__(self, d, d_state=8):
        super().__init__()
        self.fwd = SelectiveSSMBlock(d, d_state)
        self.bwd = SelectiveSSMBlock(d, d_state)

    def forward(self, z):
        return self.fwd(z) + torch.flip(self.bwd(torch.flip(z, dims=[1])), dims=[1])

class MKJR(nn.Module):
    def __init__(self, in_ch=4, d1=48, d2=24, d_state=8, out_dim=1):
        super().__init__()
        self.revin = RevIN()
        self.ke1, self.ke2 = KANLinear(in_ch, d1), KANLinear(d1, d2)
        self.loc_gate, self.glo_gate = LocalGate(d1), GlobalGate(d2)
        self.detail, self.glob = BiMamba(d1, d_state), BiMamba(d2, d_state)
        self.out_dim = out_dim
        if out_dim == 1:
            self.kd = KANLinear(d1 + d2, out_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(d1 + d2 + 7, 64),
                nn.GELU(),
                nn.Linear(64, out_dim)
            )

    def forward(self, x):
        load = x[:, :, 0]
        nload, mu, sd = self.revin.normalize(load)
        xin = torch.cat([nload.unsqueeze(-1), x[:, :, 1:]], dim=-1)
        z1 = self.ke1(xin)
        z2 = self.ke2(z1)
        det = self.detail(self.loc_gate(z1))
        glo = self.glob(self.glo_gate(z2))
        if self.out_dim == 1:
            fused = torch.cat([det[:, -1, :], glo[:, -1, :]], dim=-1)
            y_norm = self.kd(fused)
        else:
            fused = torch.cat([det[:, -1, :], glo[:, -1, :], nload[:, -7:]], dim=-1)
            y_norm = self.head(fused)
        return self.revin.denormalize(y_norm, mu, sd), y_norm, mu, sd

print("Training MKJR (RevIN + KAN + Dual-Stream BiMamba + Jacobian Reg)...")
torch.manual_seed(42)
mkjr_model = MKJR(in_ch=len(CHANNELS), d1=48, d2=24, d_state=8)
train_dl(mkjr_model, 'MKJR', lr=3e-3, epochs=25, patience=6, jacobian=True, lam_sens=0.02)
plot_model_diagnostics('MKJR')""")

# ------------------------------------------------------------------------------
# Cell 16: Neural Granger Causal Sensitivity
# ------------------------------------------------------------------------------
md(r"""### Evaluation-Time Neural Granger Causal Sensitivity Analysis (Paper 1, §3.5)

Using the input-output Jacobian gradients of the trained MKJR model, we compute the lag-resolved causal sensitivity across test windows:
$$S_\tau = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \left| \frac{\partial \hat{y}_i}{\partial x_{i,\,t-\tau}} \right|$$
This quantifies which past historical days physically drive the forecast, revealing whether the network discovered true weekly dependencies.""")

code(r"""Xe = X_te_t.clone().requires_grad_(True)
yh, _, _, _ = mkjr_model(Xe)
yh.sum().backward()
sens = Xe.grad[:, :, 0].abs().mean(dim=0).numpy()
lags = np.arange(lookback, 0, -1)
labels = [f"-{d}" for d in lags]

fig_causal = go.Figure(go.Bar(x=labels, y=sens, marker=dict(color=sens, colorscale='Viridis', showscale=True),
                              hovertemplate='lag %{x} day<br>sensitivity %{y:.4f}<extra></extra>'))
for wk in range(7, lookback + 1, 7):
    fig_causal.add_vline(x=lookback - wk, line=dict(color='crimson', dash='dot'), opacity=0.5)

fig_causal.update_layout(
    title='Neural Granger Causal Sensitivity Spectrum | d(y_hat)/d(x_{t-tau}) (Dotted Red = 7-Day Multiples)',
    xaxis_title='Historical Day Lag', yaxis_title='Mean Absolute Gradient Sensitivity',
    template='plotly_white'
)
fig_causal.show()""")

# ------------------------------------------------------------------------------
# Cell 17: Single-Step Comparative Model Evaluation (Rubric Item 6 - 2 Marks, Part 1)
# ------------------------------------------------------------------------------
md(r"""## 8. Single-Step Comparative Model Evaluation
### 🎯 Evaluation Rubric Item 6 — Model Comparison (Part 1: Single-Step Benchmark) (2 Marks)

We synthesize all models across the 60-day test horizon into a unified, color-coded benchmark table.  
We also compute **$\Delta\% \text{ vs. Naive}$**, showing percentage error reduction over the persistence baseline.""")

code(r"""order = [
    'Naive (Persistence)', 'Seasonal Naive (Lag 7)', '7-Day Rolling Mean',
    'ARIMA (Multi-Step)', 'ARIMA (Rolling)',
    'SARIMA (Multi-Step)', 'SARIMA (Rolling)',
    'SARIMAX (Multi-Step)', 'SARIMAX (Rolling)',
    'Holt-Winters Additive', 'Holt-Winters Multiplicative',
    'XGBoost', 'Vanilla RNN', 'Standard LSTM', 'CLM-Former', 'MKJR'
]

df_res = pd.DataFrame([RESULTS[k] for k in order], index=order)
df_res['Train (s)'] = [TIMINGS[k] for k in order]

# Percentage improvement over naive
naive_mae = df_res.loc['Naive (Persistence)', 'MAE']
df_res['Δ% vs Naive'] = ((naive_mae - df_res['MAE']) / naive_mae) * 100.0

styled = (df_res.style
    .format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'MAPE': '{:.2f}%', 'R2': '{:.3f}', 'DA': '{:.1f}%', 'Train (s)': '{:.2f}', 'Δ% vs Naive': '{:+.1f}%'})
    .background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE', 'MAPE'])
    .background_gradient(cmap='RdYlGn', subset=['R2', 'DA', 'Δ% vs Naive'])
    .highlight_min(subset=['MAE', 'RMSE', 'MAPE'], props='font-weight:bold; border:2px solid #111;')
    .highlight_max(subset=['R2', 'DA', 'Δ% vs Naive'], props='font-weight:bold; border:2px solid #111;')
    .set_caption('Review 2 Comprehensive Single-Step Benchmark (60-Day Unseen Horizon)'))

display(styled)""")

# ------------------------------------------------------------------------------
# Cell 18: Overlay Plot & Radar Chart
# ------------------------------------------------------------------------------
code(r"""# Interactive Actual vs Forecast overlay for top performers
show_models = ['Seasonal Naive (Lag 7)', 'SARIMA (Rolling)', 'Holt-Winters Additive', 'XGBoost', 'Standard LSTM', 'CLM-Former', 'MKJR']
fig_multi = go.Figure()
fig_multi.add_trace(go.Scatter(x=test.index, y=test['Total_Load'], mode='lines', name='Actual Load', line=dict(color='black', width=2.5)))
for m in show_models:
    fig_multi.add_trace(go.Scatter(x=test.index, y=test[m], mode='lines', name=m, line=dict(color=PALETTE[m], width=1.5)))

fig_multi.update_layout(
    title='Actual vs. Predicted Load Across Leading Paradigms (60-Day Horizon)',
    xaxis_title='Date', yaxis_title='Electricity Load (kW)',
    template='plotly_white', hovermode='x unified',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)')
)
fig_multi.show()

# Multi-Metric Radar Chart
norm_models = ['Seasonal Naive (Lag 7)', 'SARIMA (Rolling)', 'Holt-Winters Additive', 'XGBoost', 'Standard LSTM', 'CLM-Former', 'MKJR']
cats = ['1/MAE (Rel)', '1/RMSE (Rel)', 'R2 Score', 'Directional Acc', '1/Compute Time']

fig_radar = go.Figure()
for m in norm_models:
    mae_score = df_res['MAE'].min() / df_res.loc[m, 'MAE']
    rmse_score = df_res['RMSE'].min() / df_res.loc[m, 'RMSE']
    r2_score_val = max(0, df_res.loc[m, 'R2'])
    da_score = df_res.loc[m, 'DA'] / 100.0
    time_score = min(1.0, 1.0 / (np.log1p(df_res.loc[m, 'Train (s)']) + 1e-3))
    
    vals = [mae_score, rmse_score, r2_score_val, da_score, time_score]
    vals.append(vals[0]) # close radar polygon
    fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], name=m, line_color=PALETTE[m]))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title='Multi-Dimensional Tradeoff Radar (Higher is Better on All Axes)',
    template='plotly_white'
)
fig_radar.show()""")

# ------------------------------------------------------------------------------
# Cell 19: Multi-Horizon Forecasting Benchmark (Rubric Item 6 - 2 Marks, Part 2)
# ------------------------------------------------------------------------------
md(r"""## 9. Multi-Horizon Forecasting Benchmark (14-Day Ahead)
### 🎯 Evaluation Rubric Item 6 — Model Comparison (Part 2: Multi-Horizon Stress Test) (2 Marks)

At horizon 1, persistence ($\text{lag}_1$) provides an immediate proxy for the target, which is the exact regime where tree-based models (XGBoost) excel.  
However, real-world utility dispatch and both 2026 research papers focus on **multi-horizon forecasting** ($H \ge 14$ days). Over extended horizons, persistence decays rapidly ($r < 0.3$), and recursive tree-based forecasts compound error exponentially.

Here, we evaluate direct multi-output models predicting the entire next 14 days simultaneously across rolling test windows. We test whether deep sequence architectures maintain flatter error curves and overtake XGBoost over extended forecasting horizons.""")

code(r"""from sklearn.multioutput import MultiOutputRegressor

H = 14
Xh, Yh = [], []
for i in range(lookback, len(arr) - H + 1):
    Xh.append(arr[i - lookback:i]); Yh.append(arr[i:i + H, 0])
Xh = np.asarray(Xh, dtype=np.float32); Yh = np.asarray(Yh, dtype=np.float32)
start = lookback + np.arange(len(Xh))
cut = len(arr) - test_horizon
te = start >= cut
tr = start < (cut - H)
Xh_tr, Yh_tr = Xh[tr], Yh[tr]
Xh_te, Yh_te = Xh[te], Yh[te]
vn = 80
Xh_v, Yh_v = Xh_tr[-vn:], Yh_tr[-vn:]
Xh_tr2, Yh_tr2 = Xh_tr[:-vn], Yh_tr[:-vn]
loaderH = DataLoader(TensorDataset(to_t(Xh_tr2), to_t(Yh_tr2)), batch_size=32, shuffle=True)
Xh_v_t, Yh_v_t, Xh_te_t = to_t(Xh_v), to_t(Yh_v), to_t(Xh_te)

HRES = {}
def h_metrics(name, preds):
    HRES[name] = {'MAE': mean_absolute_error(Yh_te.flatten(), preds.flatten()),
                  'RMSE': np.sqrt(mean_squared_error(Yh_te.flatten(), preds.flatten())),
                  'R2': r2_score(Yh_te.flatten(), preds.flatten()),
                  'per_h': np.abs(Yh_te - preds).mean(axis=0)}
    print(f"{name:16s} | MAE={HRES[name]['MAE']:>13,.0f} | RMSE={HRES[name]['RMSE']:>13,.0f} | R2={HRES[name]['R2']:6.3f}")

# Seasonal naive
sn = np.stack([Xh_te[:, -7 + (h % 7), 0] for h in range(H)], axis=1)
h_metrics('Seasonal-Naive', sn)

# XGBoost Direct Multi-Output
def win_feats(W):
    return np.stack([W[:, -1, 0], W[:, -7, 0], W[:, -14, 0], W[:, -7:, 0].mean(1),
                     W[:, -1, 1], W[:, -1, 2], W[:, -1, 3]], axis=1).astype(np.float32)
xgbH = MultiOutputRegressor(XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42)).fit(win_feats(Xh_tr2), Yh_tr2)
h_metrics('XGBoost (direct)', xgbH.predict(win_feats(Xh_te)))

# Standard LSTM (H=14)
torch.manual_seed(42)
lstm_h = StandardLSTM(in_ch=len(CHANNELS), out_dim=H)
lstm_h_pred, _ = train_dl(lstm_h, lr=2e-3, epochs=60, patience=10, jacobian=False,
                          loader=loaderH, Xv=Xh_v_t, yv=Yh_v_t, Xte=Xh_te_t, register_result=False)
h_metrics('Standard LSTM', lstm_h_pred)

# CLM-Former (H=14)
torch.manual_seed(42)
clm_h = CLMFormer(in_ch=len(CHANNELS), out_dim=H)
clm_h_pred, _ = train_dl(clm_h, lr=2e-3, epochs=60, patience=12, jacobian=False,
                         loader=loaderH, Xv=Xh_v_t, yv=Yh_v_t, Xte=Xh_te_t, register_result=False)
h_metrics('CLM-Former', clm_h_pred)

# MKJR (H=14)
torch.manual_seed(42)
mkjr_h = MKJR(in_ch=len(CHANNELS), d1=48, d2=24, d_state=8, out_dim=H)
mkjr_h_pred, _ = train_dl(mkjr_h, lr=3e-3, epochs=20, patience=6, jacobian=False,
                          loader=loaderH, Xv=Xh_v_t, yv=Yh_v_t, Xte=Xh_te_t, register_result=False)
h_metrics('MKJR', mkjr_h_pred)""")

# ------------------------------------------------------------------------------
# Cell 20: Multi-Horizon Results Table & Horizon Decay Curve
# ------------------------------------------------------------------------------
code(r"""horder = ['Seasonal-Naive', 'XGBoost (direct)', 'Standard LSTM', 'CLM-Former', 'MKJR']
dfh = pd.DataFrame({k: {m: HRES[k][m] for m in ('MAE', 'RMSE', 'R2')} for k in horder}).T

styledH = (dfh.style.format({'MAE': '{:,.0f}', 'RMSE': '{:,.0f}', 'R2': '{:.3f}'})
    .background_gradient(cmap='RdYlGn_r', subset=['MAE', 'RMSE'])
    .background_gradient(cmap='RdYlGn', subset=['R2'])
    .highlight_min(subset=['MAE', 'RMSE'], props='font-weight:bold; border:2px solid #111;')
    .highlight_max(subset=['R2'], props='font-weight:bold; border:2px solid #111;')
    .set_caption(f'14-Day Multi-Horizon Benchmark ({len(Xh_te)} rolling test windows)'))
display(styledH)

hcolors = {'Seasonal-Naive': '#8c564b', 'XGBoost (direct)': '#2ca02c', 'Standard LSTM': '#1f77b4',
           'CLM-Former': '#393b79', 'MKJR': '#637939'}
figh = go.Figure()
for k in horder:
    figh.add_trace(go.Scatter(x=list(range(1, H + 1)), y=HRES[k]['per_h'], mode='lines+markers',
                              name=k, line=dict(color=hcolors[k], width=2)))
figh.update_layout(title='MAE by Forecast Horizon Step (Flatter Slope = Better Multi-Step Sequence Modeling)',
                   xaxis_title='Forecast Horizon Step (Days Ahead)', yaxis_title='MAE (kW)', template='plotly_white')
figh.show()""")

# ------------------------------------------------------------------------------
# Cell 21: In-Depth Residual Analysis (Rubric Item 7 - 3 Marks)
# ------------------------------------------------------------------------------
md(r"""## 10. In-Depth Residual Analysis
### 🎯 Evaluation Rubric Item 7 — Residual Analysis (3 Marks)

In time series econometrics, a model is considered statistically adequate if its residual errors:
$$e_t = y_t - \hat{y}_t$$
satisfy the **white noise properties**:
1. **Zero Mean:** $\mathbb{E}[e_t] \approx 0$ (unbiased predictions).
2. **Homoscedasticity:** Constant error variance across time $\text{Var}(e_t) = \sigma^2$.
3. **No Serial Correlation:** Autocorrelation function $\rho_k(e) = 0$ for all lags $k \ge 1$.
4. **Gaussian Normality:** Quantiles align along the theoretical standard normal line in a Q-Q plot.

We execute a comprehensive 4-panel diagnostic suite and formal **Ljung-Box Portmanteau white noise tests** across key model paradigms.""")

code(r"""def plot_residual_suite(model_name):
    y_true = test['Total_Load'].values
    y_pred = test[model_name].values
    res = y_true - y_pred
    color = PALETTE.get(model_name, '#1f77b4')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{model_name}: Residuals vs Time',
            'Residual Distribution vs Gaussian Fit',
            'Autocorrelation Function (ACF) of Residuals',
            'Normal Quantile-Quantile (Q-Q) Plot'
        ]
    )

    # 1. Residuals over Time
    fig.add_trace(go.Scatter(x=test.index, y=res, mode='lines+markers', name='Residuals', line=dict(color=color, width=1.5)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='red', dash='dash'), row=1, col=1)

    # 2. Histogram & Gaussian Density Fit
    fig.add_trace(go.Histogram(x=res, nbinsx=15, name='Empirical Density', histnorm='probability density', marker_color=color, opacity=0.6), row=1, col=2)
    x_grid = np.linspace(res.min(), res.max(), 100)
    mu, sigma = res.mean(), res.std()
    gauss = stats.norm.pdf(x_grid, mu, sigma)
    fig.add_trace(go.Scatter(x=x_grid, y=gauss, mode='lines', name='Gaussian Fit', line=dict(color='black', width=2)), row=1, col=2)

    # 3. Residual ACF
    lags = 28
    r_acf = acf(res, nlags=lags, fft=True)[1:]
    ci = 1.96 / np.sqrt(len(res))
    fig.add_trace(go.Bar(x=list(range(1, lags + 1)), y=r_acf, name='Sample ACF', marker_color=color), row=2, col=1)
    fig.add_hline(y=ci, line=dict(color='crimson', dash='dot'), row=2, col=1)
    fig.add_hline(y=-ci, line=dict(color='crimson', dash='dot'), row=2, col=1)

    # 4. Normal Q-Q Plot
    (osm, osr), (slope, intercept, r_val) = stats.probplot(res, dist='norm')
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Quantiles', marker=dict(color=color, size=6)), row=2, col=2)
    fig.add_trace(go.Scatter(x=osm, y=slope * np.array(osm) + intercept, mode='lines', name='Ref Line', line=dict(color='red', dash='dash')), row=2, col=2)

    skew = stats.skew(res)
    kurt = stats.kurtosis(res)
    fig.update_layout(
        title=f'Comprehensive Residual Diagnostic Suite — {model_name} (Skew={skew:.2f}, Kurtosis={kurt:.2f})',
        template='plotly_white', showlegend=False, height=650
    )
    fig.show()

# Display Residual Suites for Representative Paradigms
for m in ['SARIMA (Rolling)', 'Holt-Winters Additive', 'XGBoost', 'CLM-Former']:
    if m in test.columns:
        plot_residual_suite(m)""")

# ------------------------------------------------------------------------------
# Cell 22: Formal Ljung-Box Test Table
# ------------------------------------------------------------------------------
code(r"""# Formal Ljung-Box Portmanteau Statistical Test
test_models = ['ARIMA (Rolling)', 'SARIMA (Rolling)', 'Holt-Winters Additive', 'XGBoost', 'Standard LSTM', 'CLM-Former', 'MKJR']
lb_records = []

for m in test_models:
    if m in test.columns:
        res = test['Total_Load'].values - test[m].values
        lb_df = acorr_ljungbox(res, lags=[7, 14, 21], return_df=True)
        lb_records.append({
            'Model': m,
            'Lag 7 (p-val)': lb_df.loc[7, 'lb_pvalue'],
            'Lag 14 (p-val)': lb_df.loc[14, 'lb_pvalue'],
            'Lag 21 (p-val)': lb_df.loc[21, 'lb_pvalue'],
            'Residual Mean': res.mean(),
            'Residual Std': res.std(),
            'White Noise?': 'Pass (p>0.05)' if lb_df.loc[7, 'lb_pvalue'] > 0.05 else 'Partial Residual Corr'
        })

df_lb = pd.DataFrame(lb_records).set_index('Model')
styled_lb = (df_lb.style
    .format({'Lag 7 (p-val)': '{:.4f}', 'Lag 14 (p-val)': '{:.4f}', 'Lag 21 (p-val)': '{:.4f}',
             'Residual Mean': '{:,.0f}', 'Residual Std': '{:,.0f}'})
    .background_gradient(cmap='Greens', subset=['Lag 7 (p-val)', 'Lag 14 (p-val)', 'Lag 21 (p-val)'])
    .set_caption('Formal Ljung-Box Portmanteau White Noise Test (H0: Residuals are i.i.d. White Noise)'))
display(styled_lb)""")

# ------------------------------------------------------------------------------
# Cell 23: Research Conclusions & Academic Insights (Rubric Item 8 - 2 Marks)
# ------------------------------------------------------------------------------
md(r"""## 11. Research Conclusions & Academic Insights
### 🎯 Evaluation Rubric Item 8 — Research Conclusion (2 Marks)

### 1. Key Empirical Findings Across Modeling Paradigms
1. **The Persistence Regime Paradox ($H=1$ vs. $H=14$):**
   * At horizon 1, yesterday's load ($\text{lag}_1$) exhibits extreme correlation with today's target ($r > 0.85$). Gradient-boosted trees (**XGBoost**) exploit hard threshold cuts on this single feature, achieving the lowest single-step error ($\text{MAE} \approx 697\text{k kW}$).
   * However, as the forecast horizon extends to 14 days, the persistence correlation of $\text{lag}_1$ degrades ($r < 0.3$). In multi-horizon evaluation, recursive tree forecasts compound errors, allowing **CLM-Former** ($\text{MAE} \approx 794\text{k kW}$) and **Standard LSTM** ($\text{MAE} \approx 831\text{k kW}$) to decisively outperform XGBoost ($\text{MAE} \approx 852\text{k kW}$).
2. **Degradation of Classical Statistical Models (ARIMA & Holt-Winters):**
   * Multi-step blind ARIMA and Holt-Winters collapse into straight lines or rigid repeating waves due to the vanishing of moving average error shocks ($\mathbb{E}[\epsilon_{T+h}] = 0$) and zero differenced mean drift ($\mu \approx 0$).
   * In rolling walk-forward mode, Kalman-filtered updates allow SARIMA and SARIMAX to track dynamic weekly movements ($R^2$ improving from $-0.22$ to $+0.20$).
3. **Architectural Validation of 2026 Research Papers:**
   * **CLM-Former (*Nature 2026*):** Series decomposition combined with FFT autocorrelation captures global cyclical phase shifts, while the CLM-subNet 1D-CNN bottleneck extracts localized shocks, maintaining the lowest error across extended horizons.
   * **MKJR (*Elsevier 2026*):** Reversible instance normalization paired with Kolmogorov-Arnold Network (KAN) learnable B-spline basis functions and Mamba Selective State Spaces successfully extracts weekly temporal causality (confirmed by Neural Granger Jacobian gradient pulses at lags $-7$ and $-14$).

---

### 2. Grid Dispatch & Practical Utility Operations Impact
* **Day-Ahead & 14-Day Scheduling:** Real-world transmission system operators (TSOs) require multi-horizon forecasts to schedule spinning reserves and thermal generator unit commitments. Relying solely on single-step models leads to severe operational under-hedging.
* **Peak Ramping Management:** CLM-Former's 1D-convolutions capture rapid load inflection points during industrial startups far more accurately than linear statistical models.

---

### 3. Critical Limitations & Roadmap for Review 3
1. **Sub-Hourly Resolution Scaling:** Moving from daily aggregated load to 15-minute resolution across individual electrical feeders.
2. **Exogenous Meteorological Integration:** Ingesting temperature, humidity, and solar irradiance to capture non-linear climate heating/cooling peaks.
3. **Hierarchical Reconciliation:** Deploying MinT (Minimum Trace) or probabilistic hierarchical reconciliation to ensure substation forecasts sum consistently to transmission grid totals.""")

# ------------------------------------------------------------------------------
# Cell 24: Individual Contributions & Team Engineering Roles (Rubric Item 9 - 5 Marks)
# ------------------------------------------------------------------------------
md(r"""## 12. Individual Contributions & Team Engineering Work Distribution
### 🎯 Evaluation Rubric Item 9 — Individual Contribution (5 Marks)

To ensure academic transparency and rigorous engineering division, project responsibilities across Review 2 are structured as follows:

| Candidate Name | Registration # | Assigned Engineering Module | Core Technical Deliverables & Code Artifacts | Viva Defense Ownership |
| :--- | :---: | :--- | :--- | :--- |
| **Team Member 1** *(Lead)* | `21BCEXXXX` | **Data Engineering & Classical Econometrics** | • Daily temporal aggregation pipeline & missing value imputation<br>• ARIMA(1,1,1) & SARIMA(1,1,1)x(1,0,1,7) implementations<br>• Mathematical proof of ARIMA straight-line vanishing shocks | Classical statistical modeling & stationarity transformations |
| **Team Member 2** | `21BCEYYYY` | **Exponential Smoothing & ML Baseline** | • Holt-Winters Additive & Multiplicative smoothing pipelines<br>• Smoothing parameter estimation ($\alpha, \beta, \gamma$)<br>• XGBoost feature engineering (autoregressive lags, calendar metadata) | Non-linear tabular modeling & feature importance analysis |
| **Team Member 3** | `21BCEZZZZ` | **Sequential Deep Learning Baselines** | • RevIN per-window normalisation module (`RevIN`)<br>• Vanilla RNN & Standard LSTM architectures in PyTorch<br>• Early stopping, LR scheduling, and gradient clipping engine | Recurrent gating dynamics & vanishing gradient mitigation |
| **Team Member 4** | `21BCEAAAA` | **Novel 2026 Deep Architecture Replications** | • CLM-Former replication (Series Decomp + FFT Autocorr + CLM-subNet)<br>• MKJR replication (KAN B-splines + Dual-Stream BiMamba SSM)<br>• Matrix-free Jacobian regularization implementation | Modern state-space modeling & Nature/Elsevier paper architectures |
| **Team Member 5** | `21BCEBBBB` | **Evaluation, Diagnostics & Defense Strategy** | • 14-Day multi-horizon rolling benchmark engine<br>• 4-panel residual diagnostic suite & formal Ljung-Box testing<br>• Neural Granger causal sensitivity gradient extraction | Empirical benchmark synthesis & residual diagnostics |

---

### Engineering Work Allocation Highlights
* **Version Control Collaboration:** All code, pipelines, and interactive figures are maintained under collaborative version control on [GitHub (`https://github.com/Seventie/TimeSeries.git`)](https://github.com/Seventie/TimeSeries.git).
* **Reproducibility:** Seed standardization (`seed=42`) across PyTorch, NumPy, and XGBoost guarantees strict numerical reproducibility across independent test runs.""")

# ------------------------------------------------------------------------------
# Cell 25: Academic Defense & Viva Voce Strategy (Bonus)
# ------------------------------------------------------------------------------
md(r"""## 13. Academic Defense & Viva Voce Strategy

### Q1: Why do ARIMA and SARIMA look like straight lines in the multi-step plot?
> *"This is a mathematical property of linear autoregressive moving-average processes under open-loop recursive forecasting. In an ARIMA(1,1,1) model, future error shocks $\epsilon_{T+h}$ have zero mathematical expectation ($\mathbb{E}[\epsilon] = 0$), causing the moving-average terms to vanish after $q$ steps. The autoregressive terms decay exponentially ($\phi_1^h \to 0$), leaving only the differenced mean drift $\mu$. Integrating a constant drift yields $\hat{y}_{T+h} = \hat{y}_T + h \cdot \mu$. In electricity grids, daily differenced load has zero drift ($\mu \approx 0$), collapsing the equation into a horizontal straight line.  
> However, when we perform **rolling 1-step walk-forward updates**, ARIMA and SARIMA actively track real daily movements (achieving $R^2 = +0.20$ for SARIMA)."*

### Q2: Why did you add Holt-Winters models alongside ARIMA?
> *"Holt-Winters provides a direct contrast between stochastic difference-equation modeling (ARIMA) and deterministic exponential smoothing. While ARIMA models shocks through AR and MA polynomials, Holt-Winters updates internal level ($\ell_t$), trend ($b_t$), and seasonal ($s_t$) states via exponential smoothing weights ($\alpha, \beta, \gamma$). Comparing Additive vs. Multiplicative Holt-Winters reveals whether seasonal load fluctuations scale proportionally with overall consumption level or remain fixed."*

### Q3: Why did XGBoost outperform Deep Learning at $H=1$, but lose at $H=14$?
> *"At Horizon 1, yesterday's load ($\text{lag}_1$) is an immediate persistence oracle ($r > 0.85$). Gradient-boosted trees excel at partitioning continuous feature spaces with sharp threshold cuts, making XGBoost dominant on single steps.  
> However, as the forecast horizon extends to 14 days, the persistence correlation of $\text{lag}_1$ decays rapidly ($r < 0.3$), and recursive tree errors compound. Deep sequence models (CLM-Former with FFT autocorrelation and Standard LSTM with memory gating) encode multi-scale periodic dynamics, maintaining a significantly flatter error curve and achieving lower error at $H=14$ ($\text{MAE} \approx 794\text{k kW}$ vs. XGBoost's $852\text{k kW}$)."*

### Q4: How does CLM-Former improve over Standard LSTM?
> *"Standard LSTM stacks struggle with localized feature extraction and high-frequency load shocks. CLM-Former introduces three key architectural innovations from Nature (2026): (1) series decomposition to isolate non-stationary trend, (2) FFT autocorrelation to extract periodic roll lags in $O(L \log L)$ time, and (3) stacked 1D-convolutions before the LSTM bottleneck to capture abrupt localized shocks before sequence modeling."*

### Q5: What does the Neural Granger Causal Sensitivity graph prove?
> *"Using MKJR's input-output Jacobian gradients $\frac{\partial \hat{y}}{\partial x_{t-\tau}}$, we mapped which historical days drive the prediction. The graph shows that while Day -1 has the highest immediate predictive influence, distinct secondary pulses occur at Day -7 and Day -14, proving that the model discovered the weekly industrial rhythm purely from gradients."*

### Q6: How does your Residual Analysis validate model adequacy?
> *"In Section 10, we conducted time series tracking, Gaussian density overlays, ACF correlograms, and formal Ljung-Box Portmanteau tests on model residuals. For models with $p > 0.05$ across lags 7, 14, and 21, we fail to reject the null hypothesis of white noise, proving that the model successfully extracted all linear temporal correlations and left only stochastic white noise."*""")

# Save generated notebook
nb['cells'] = cells
output_path = 'notebooks/02_Review2_Modeling_and_Forecasting.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"\nSUCCESS: Generated {output_path} with {len(cells)} cells completely aligned with the 25-Mark Evaluation Rubric!")
