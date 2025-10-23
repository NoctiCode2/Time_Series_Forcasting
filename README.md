# ⚡ Electricity Transformer Temperature & Exchange Rate Forecasting

### *By* Thanina Ait Ferhat, Kamilia Benlamara, Iness Fendes & Lynda Benkerrou  
*Université Paris Cité – 2024*

---

## 🧠 Project Overview

This project focuses on **time series forecasting** applied to two distinct problems:

1. **Univariate Forecasting** – Predicting transformer oil temperature (OT) using the **ETTh1 dataset**.  
2. **Multivariate Forecasting** – Predicting exchange rate fluctuations across eight countries using the **Exchange Rate dataset**.

We explore and compare advanced **machine learning models**:
- **LSTM (Long Short-Term Memory)** networks for univariate temporal prediction.
- **Random Forest Regressor** for multivariate regression tasks.

The goal is to assess each model’s ability to handle **temporal dependencies**, **missing data**, and **non-linear relationships** while achieving strong predictive accuracy.

---

## 📁 Repository Structure

```
├── Code_Part_A.ipynb         # LSTM model for univariate forecasting (ETTh1 dataset)
├── Code_Part_B.ipynb         # Random Forest model for multivariate forecasting (Exchange Rate dataset)
├── AitFerhat_Benlamara_Benkerrou_Fendes.pdf  # Full project report
├── README.md                 # Project documentation
```

---

## 🔍 Datasets

### 1️⃣ ETTh1 Dataset – *Electricity Transformer Temperature*
- Source: Electricity transformer dataset from July 2016 to July 2018 (China)
- Variables:
  - `date`: Timestamp of measurement  
  - `OT`: Transformer Oil Temperature  
- Characteristics:
  - 17,420 observations
  - No missing values

### 2️⃣ Exchange Rate Dataset
- Source: Exchange rates for 8 countries (1990–2010)
- Variables: Daily exchange rates (8 columns)
- Characteristics:
  - 7,588 observations
  - Contains missing values (handled via spline interpolation)

---

## ⚙️ Methods & Models

### 🔸 Part A – LSTM (Univariate Forecasting)
We used a **Stacked LSTM** architecture to predict 100 future values of the OT variable.

#### Preprocessing
- Date conversion and normalization (MinMax scaling)
- Sliding window approach (7 time steps)
- 99% training / 1% test split

#### Training
- Optimizer: `Adam`
- Loss: `Mean Absolute Error (MAE)`
- Early stopping applied
- Parameter tuning (47 experiments via Kaggle)

#### Best Model Parameters
| Hyperparameter | Value |
|----------------|--------|
| Units | 100 |
| Time steps | 7 |
| Dropout | 0.1 |
| Epochs | 20 |

**Best Result:** MAE = **0.65**

---

### 🔸 Part B – Random Forest Regressor (Multivariate Forecasting)
We used a **RandomForestRegressor** to predict 100 future values of the target variable `'6'`.

#### Preprocessing
- Missing value handling via spline interpolation (order 5)
- Lag features added to capture temporal dependencies
- 90% training / 10% testing split

#### Training
- Tuned hyperparameters: number of trees and tree depth
- 119 experiments conducted to optimize performance

#### Best Model Parameters
| Trees | Depth | MAE |
|--------|--------|------|
| 385 | 22 | **0.00941** |

---

## 📊 Results Summary

| Task | Model | Type | Best MAE | Key Insight |
|------|--------|------|-----------|--------------|
| Transformer Temperature | LSTM | Univariate | 0.65 | LSTM captures long-term dependencies well |
| Exchange Rate Forecasting | Random Forest | Multivariate | 0.00941 | Ensemble model handles noise & correlations effectively |

---

## 🧩 Key Takeaways

- **LSTMs** are powerful for modeling sequential dependencies but require careful tuning to avoid overfitting.  
- **Random Forests** excel in robustness and generalization, especially with proper feature engineering.  
- Strategic **data preprocessing** (handling missing values, scaling, lag features) is critical for performance.  

---

## 🚀 Future Work

- Experiment with **hybrid models** combining LSTM and ensemble methods.  
- Integrate **additional contextual variables** (e.g., weather, market factors).  
- Explore **attention-based architectures** (Transformers) for improved forecasting accuracy.  

---

## 🧰 Requirements

To reproduce the results, install the following dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

Run the notebooks in order:
1. `Code_Part_A.ipynb` → Univariate Forecasting  
2. `Code_Part_B.ipynb` → Multivariate Forecasting  

---

## 🧾 Citation

If you use this work, please cite:

> Ait Ferhat, T., Benlamara, K., Fendes, I., & Benkerrou, L. (2024).  
> *Electricity Transformer Temperature and Exchange Rate Forecasting.*  
> University of Paris Cité.
