import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# Phase 2: Predictive Modeling and Forecasting (Review 2)
### Decoding Electricity Demand: Analyzing Consumption Patterns and Forecasting Future Energy Load

In this notebook, we move from Exploratory Data Analysis (EDA) into Predictive Modeling. Based on our findings from Phase 1, we established that the aggregated power grid load is non-stationary but becomes stationary after 1st-order differencing, and it contains strong 7-day (weekly) and 365-day (annual) seasonalities.

For **Review 2**, we implement the following progression of forecasting models:
1. **ARIMA**: A standard statistical baseline (ignoring seasonality).
2. **SARIMA**: A seasonal statistical model incorporating the 7-day weekly cycle.
3. **XGBoost**: A powerful Machine Learning baseline using lag and calendar features.
4. **CNN-LSTM Hybrid (CLM-subNet inspired)**: A novel Deep Learning architecture inspired by recent literature.

### Literature Integration
We integrate architectural concepts from the recent paper:
> *"CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model"* (PMC12867978).

The paper proposes a novel framework where the time series is decomposed into Trend and Seasonal components. While the Trend is processed linearly, the Seasonal component is passed through a **CLM-subNet**—a specific architecture consisting of a **1D-Convolutional Neural Network (CNN)** followed by a **Long Short-Term Memory (LSTM)** network. The CNN captures high-frequency, localized, aperiodic fluctuations (like sudden residential load changes), while the LSTM models the longer-term sequential dependencies of these extracted features.

We will explicitly implement this **CNN-LSTM Hybrid** paradigm in PyTorch and compare its performance against the statistical and ML baselines."""))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Statistical & ML Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
"""))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Preparation
To make the baseline statistical models (ARIMA/SARIMA) computationally tractable on standard hardware, we will aggregate the 15-minute interval data to **Daily** resolution. Modeling 15-minute data with SARIMA requires estimating thousands of parameters which is highly inefficient. We will focus on predicting the aggregated daily grid load."""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset
print("Loading data... (this may take a minute)")
df = pd.read_csv('../data/LD2011_2014.txt', sep=';', decimal=',', 
                 parse_dates=[0], index_col=0, low_memory=False)

# Aggregate all 370 clients to get total grid load
grid_load = df.sum(axis=1)

# Resample to Daily frequency
daily_load = grid_load.resample('D').sum()

# Filter data to start from 2012 (2011 has many missing/zero values for clients)
daily_load = daily_load.loc['2012-01-01':]

# Convert to DataFrame for easier manipulation
df_daily = pd.DataFrame({'Total_Load': daily_load})

print(f"Daily Data Shape: {df_daily.shape}")
df_daily.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Train-Test Split
# We will use the last 60 days for testing our models
test_days = 60
train = df_daily.iloc[:-test_days].copy()
test = df_daily.iloc[-test_days:].copy()

print(f"Training Set: {len(train)} days")
print(f"Testing Set:  {len(test)} days")

plt.figure(figsize=(14, 5))
plt.plot(train.index, train['Total_Load'], label='Train')
plt.plot(test.index, test['Total_Load'], label='Test', color='orange')
plt.title('Train / Test Split (Aggregated Daily Load)')
plt.ylabel('Electricity Load (kW)')
plt.legend()
plt.show()
"""))

# Baseline 1: ARIMA
cells.append(nbf.v4.new_markdown_cell("""## 2. Statistical Baseline: ARIMA
First, we fit a standard ARIMA model. From Phase 1, we know the data requires $d=1$ differencing. We use a simple ARIMA(1,1,1) model."""))

cells.append(nbf.v4.new_code_cell("""%%time
# Fit ARIMA(1, 1, 1)
print("Training ARIMA...")
arima_model = ARIMA(train['Total_Load'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast
arima_forecast = arima_result.forecast(steps=len(test))
test['ARIMA_Forecast'] = arima_forecast.values

print(arima_result.summary().tables[1])
"""))

# Baseline 2: SARIMA
cells.append(nbf.v4.new_markdown_cell("""## 3. Seasonal Baseline: SARIMA
Because electricity demand has a massive drop on weekends (as factories close), a 7-day seasonality is strictly required. We implement a SARIMA(1,1,1)x(1,0,1,7) model."""))

cells.append(nbf.v4.new_code_cell("""%%time
# Fit SARIMA with weekly seasonality (m=7)
print("Training SARIMA...")
# We use seasonal difference D=0 here on daily data, with AR and MA seasonal components.
sarima_model = SARIMAX(train['Total_Load'], 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 0, 1, 7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)

# Forecast
sarima_forecast = sarima_result.forecast(steps=len(test))
test['SARIMA_Forecast'] = sarima_forecast.values

print(sarima_result.summary().tables[1])
"""))

# Baseline 3: XGBoost
cells.append(nbf.v4.new_markdown_cell("""## 4. Machine Learning Baseline: XGBoost
Next, we frame the time series forecasting problem as a supervised machine learning task. We extract features such as lag values (looking back 1 day, 7 days) and calendar metadata (day of week, is_weekend)."""))

cells.append(nbf.v4.new_code_cell("""# Feature Engineering for XGBoost
def create_features(df):
    df_feat = df.copy()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    
    # Lag features
    df_feat['lag_1'] = df_feat['Total_Load'].shift(1)
    df_feat['lag_7'] = df_feat['Total_Load'].shift(7)
    
    # Rolling mean
    df_feat['rolling_mean_7'] = df_feat['Total_Load'].shift(1).rolling(window=7).mean()
    
    return df_feat.dropna()

df_xgb = create_features(df_daily)

# Split again after dropping NAs
train_xgb = df_xgb.iloc[:-test_days]
test_xgb = df_xgb.iloc[-test_days:]

X_train = train_xgb.drop('Total_Load', axis=1)
y_train = train_xgb['Total_Load']
X_test = test_xgb.drop('Total_Load', axis=1)
y_test = test_xgb['Total_Load']

print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train, y_train)

test['XGB_Forecast'] = xgb_model.predict(X_test)
"""))

# Deep Learning: CNN-LSTM
cells.append(nbf.v4.new_markdown_cell("""## 5. Novel DL Architecture: CNN-LSTM Hybrid (CLM-subNet Inspired)

Inspired by the literature, we will build a PyTorch model that utilizes a 1D-CNN followed by an LSTM. 

### Architecture Flow:
1. **Input Sequence**: We use a lookback window of 14 days to predict the next day.
2. **1D-CNN (Feature Extractor)**: Convolves over the time steps to extract high-frequency local features (e.g., sudden spikes).
3. **LSTM (Sequence Modeling)**: Takes the CNN feature maps and learns the sequential temporal dependencies.
4. **Dense Layer**: Outputs the final prediction.
"""))

cells.append(nbf.v4.new_code_cell("""# Data Preparation for PyTorch
lookback = 14

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_daily[['Total_Load']])

X_dl, y_dl = [], []
for i in range(lookback, len(scaled_data)):
    X_dl.append(scaled_data[i-lookback:i])
    y_dl.append(scaled_data[i])

X_dl, y_dl = np.array(X_dl), np.array(y_dl)

# Split
X_train_dl = X_dl[:-test_days]
y_train_dl = y_dl[:-test_days]
X_test_dl = X_dl[-test_days:]
y_test_dl = y_dl[-test_days:]

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_dl, dtype=torch.float32)
y_train_t = torch.tensor(y_train_dl, dtype=torch.float32)
X_test_t = torch.tensor(X_test_dl, dtype=torch.float32)
y_test_t = torch.tensor(y_test_dl, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
"""))

cells.append(nbf.v4.new_code_cell("""# Define the CNN-LSTM (CLM-subNet core) Model
class CNN_LSTM_Forecaster(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Forecaster, self).__init__()
        
        # 1D CNN to extract local features
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM to capture long-term dependencies of the extracted features
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        
        # Fully connected output
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        x = self.cnn(x)
        
        # LSTM expects (batch, seq_len, features)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

model = CNN_LSTM_Forecaster()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
"""))

cells.append(nbf.v4.new_code_cell("""# Training Loop
epochs = 50
print("Training CNN-LSTM...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.6f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Inference
model.eval()
with torch.no_grad():
    dl_preds = model(X_test_t).numpy()

# Inverse transform to original scale
dl_preds_unscaled = scaler.inverse_transform(dl_preds)
test['CNN_LSTM_Forecast'] = dl_preds_unscaled
"""))

# Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 6. Evaluation & Comparison
Finally, we evaluate all models using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)."""))

cells.append(nbf.v4.new_code_cell("""def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"Model": name, "MAE (kW)": mae, "RMSE": rmse, "MAPE (%)": mape}

results = []
results.append(evaluate_model(test['Total_Load'], test['ARIMA_Forecast'], "ARIMA"))
results.append(evaluate_model(test['Total_Load'], test['SARIMA_Forecast'], "SARIMA"))
results.append(evaluate_model(test['Total_Load'], test['XGB_Forecast'], "XGBoost"))
results.append(evaluate_model(test['Total_Load'], test['CNN_LSTM_Forecast'], "CNN-LSTM Hybrid"))

results_df = pd.DataFrame(results).set_index("Model")
display(results_df.round(2))
"""))

cells.append(nbf.v4.new_code_cell("""# Visualizing the final results
fig = go.Figure()

fig.add_trace(go.Scatter(x=test.index, y=test['Total_Load'], mode='lines', name='Actual Load', line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=test.index, y=test['ARIMA_Forecast'], mode='lines', name='ARIMA', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=test.index, y=test['SARIMA_Forecast'], mode='lines', name='SARIMA'))
fig.add_trace(go.Scatter(x=test.index, y=test['XGB_Forecast'], mode='lines', name='XGBoost', line=dict(color='green')))
fig.add_trace(go.Scatter(x=test.index, y=test['CNN_LSTM_Forecast'], mode='lines', name='CNN-LSTM', line=dict(color='red', width=2)))

fig.update_layout(
    title='Multi-Model Forecasting Comparison (Test Set)',
    xaxis_title='Date',
    yaxis_title='Electricity Load (kW)',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
)
fig.show()
"""))

nb['cells'] = cells

with open('notebooks/02_Review2_Modeling_and_Forecasting.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook generated successfully!")
