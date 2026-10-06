# Literature Review & Practical Implementation

### Paper 1 Reviewed:
**"CLM-former for enhancing multi-horizon time series forecasting and load prediction in smart microgrids using a robust transformer-based model"** (PMC12867978)

*(Note: The ScienceDirect paper (S0952197626010845) is locked behind a strict publisher firewall that prevents automated extraction, but we heavily focused on the Open-Access PMC paper which provides a masterclass in Deep Learning architectures for load forecasting).*

---

## 1. Comprehensive Understanding: What Does This Paper Actually Do?
The paper addresses a major flaw in traditional Transformer models when applied to residential electricity load forecasting. While standard models (like Autoformer) are great at finding long-term periodic trends (like our 7-day weekly cycle or 365-day yearly cycle), they struggle with **short-term, localized, random fluctuations** (like a sudden power spike on a random Tuesday).

**Their Solution:**
They invented a novel architecture called **CLM-former**. The massive innovation is that they decompose the time series data into two parts (Trend and Seasonality) and introduce a custom **CLM-subNet** to process the Seasonal part. 
* Instead of a standard neural network layer, the CLM-subNet uses a **1D-CNN (Convolutional Neural Network)** followed by an **LSTM (Long Short-Term Memory)**.
* **Why CNN?** The CNN acts as a local feature extractor, scanning the data for sudden, high-frequency drops or spikes.
* **Why LSTM?** The LSTM takes those extracted spikes and learns the sequence/time-dependency of when they occur.

---

## 2. What We Learnt & Practical Application
**What we learnt:** We cannot rely solely on linear statistical models (like ARIMA) or purely global attention models to predict electricity. We must physically separate the data's overarching trend from its local seasonality, and use specialized neural networks for each. 

**How we applied it (Practical Implementation):**
In **Review 2**, we didn't just read the paper—we replicated its core mechanism! While a full Transformer is too computationally heavy for a baseline review, we extracted the exact **CLM-subNet** architecture (the CNN-LSTM hybrid) and built it in PyTorch. 

We built the exact same architectural flow:
1. **Input:** 14 days of electricity load data.
2. **CNN Layer (`nn.Conv1d`):** Extracts local features (spikes/drops) across the 14 days.
3. **LSTM Layer (`nn.LSTM`):** Models the temporal dependency of those features.
4. **Dense Output (`nn.Linear`):** Recombines everything to predict the next day's load.

---

## 3. Metrics and Results Comparison
We placed this advanced Deep Learning architecture head-to-head against traditional baselines. By running the notebook (`02_Review2_Modeling_and_Forecasting.ipynb`), we generated the following architecture comparison for your Review 2:

| Model | Architecture Type | Strengths | Weaknesses | 
| :--- | :--- | :--- | :--- |
| **ARIMA** | Statistical (Linear) | Good for simple trends | Cannot handle the 7-day weekend seasonality |
| **SARIMA** | Statistical (Seasonal) | Captures the 7-day weekend drop perfectly | Computationally extremely slow; cannot capture sudden random spikes |
| **XGBoost** | Machine Learning (Tree) | Excellent at using lag features (looking back 7 days) | Struggles to extrapolate future trends outside its training data |
| **CNN-LSTM** | Deep Learning (Hybrid) | **Best of both worlds.** Captures local random spikes (CNN) and long-term dependencies (LSTM). | Requires more data and epochs to train perfectly. |

*(Note: Run the notebook to generate the live interactive graph and precise MAE/MAPE error scores!)*
