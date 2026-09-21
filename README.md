# Decoding Electricity Demand: Analyzing Consumption Patterns and Forecasting Future Energy Load

## Overview
This project aims to analyze and forecast electricity consumption patterns using the highly complex **UCI Electricity Load Diagrams (2011–2014)** dataset. The dataset comprises 15-minute interval readings across **370 distinct clients** over a 4-year period, representing a massive multivariate time-series problem.

## Objectives
1. **Exploratory Data Analysis (EDA):** Understand macro-trends, weekly seasonality, and consumer behavior by aggregating grid load.
2. **Statistical Modeling:** Prove stationarity mathematically (via Augmented Dickey-Fuller tests) and determine optimal ARIMA/SARIMA hyperparameters (p, d, q) through Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis.
3. **Machine Learning / Deep Learning Forecasting:** (Upcoming) Transition from baseline statistical models to robust multi-series Deep Learning architectures (e.g., LSTMs, Temporal Fusion Transformers) capable of independently forecasting all 370 consumers simultaneously.

## Project Structure
* `data/`: Contains the raw dataset files (ignored in git due to size).
* `notebooks/`: Contains the Jupyter Notebooks.
  * `EDA_and_Statistical_Analysis.ipynb`: The primary notebook detailing the data parsing, interactive visualizations, decomposition, and hyperparameter justifications.

## Dataset
* **Source:** UCI Machine Learning Repository
* **Name:** ElectricityLoadDiagrams20112014 Data Set
* **Characteristics:** Multivariate, Time-Series (15-min frequency, 370 attributes).

### How to Download and Setup the Dataset
Because the dataset is extremely large (~710 MB uncompressed), it is not hosted in this GitHub repository. To run the notebook locally, please follow these steps:
1. Download the dataset `.zip` file from the official UCI repository here: **[Download Link](https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip)** (250 MB).
2. Extract the `.zip` file.
3. Place the extracted `LD2011_2014.txt` file directly inside the `data/` directory of this project.
4. Run the Jupyter Notebooks.
