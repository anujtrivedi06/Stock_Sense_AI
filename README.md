# 📈 Project StockSenseAI

**Universal Sentiment Engine for Stock Price Prediction**

---

## 🧠 Overview

**Project StockSenseAI** is an end-to-end, leakage-safe machine learning system that predicts the **next trading day’s closing price** of a stock by combining:

* Historical market data
* Technical indicators
* News sentiment
* Reddit sentiment
* Google search trends

The project was built for a competitive setting and emphasizes **temporal correctness, explainability, and real-time inference**.

---

## 🎯 Objective

> Predict the **next day closing price** of a stock using **live alternative data** and historical price behavior while strictly preventing temporal leakage.

---

## 🏗️ System Architecture

```
Stock Ticker + Date Range
        │
        ▼
Data Collection
│
├── Stock Prices (yfinance)
├── News Sentiment (FinViz + Yahoo)
├── Reddit Sentiment (r/wallstreetbets, r/stocks, r/investing)
├── Google Trends
│
▼
Feature Engineering
│
├── Date Alignment
├── Missing Data Handling
├── Lagged Sentiment Features
├── Rolling Technical Indicators
│
▼
Target Definition
│
├── Next-Day Closing Price
│
▼
Model Training
│
├── Random Forest Regressor
├── Gradient Boosting Regressor
│
▼
Inference
│
├── Offline Training
├── Live Prediction Dashboard (Streamlit)
```

---

## 📊 Data Sources

### 1️⃣ Stock Market Data

* Source: **Yahoo Finance (`yfinance`)**
* Features:

  * Open, High, Low, Close, Volume
  * Technical indicators:

    * SMA (5, 20)
    * RSI (14)
    * MACD & Signal Line
    * Volatility
    * Daily Returns

---

### 2️⃣ News Sentiment

* Sources:

  * **FinViz** (primary)
  * **Yahoo Finance** (fallback)
* Method:

  * Headline scraping
  * **VADER sentiment analysis**
* Aggregation:

  * Daily average sentiment
  * Sentiment volatility
  * Positive/negative ratio
  * News volume

---

### 3️⃣ Reddit Sentiment

* Subreddits:

  * `r/wallstreetbets`
  * `r/stocks`
  * `r/investing`
* Method:

  * Reddit JSON endpoints
  * VADER sentiment on post text
* Advanced Feature:

  * **Engagement-weighted sentiment** using upvotes
* Aggregation:

  * Daily average sentiment
  * Weighted sentiment
  * Post volume
  * Engagement score

---

### 4️⃣ Google Trends

* Source: **Google Trends (pytrends)**
* Feature:

  * Normalized search interest (0–1)
* Purpose:

  * Capture public attention & curiosity spikes

---

## ⚙️ Feature Engineering (Core Component)

This is the most critical part of the project.

### 🔑 Temporal Alignment

All data sources are aligned on a **daily `Date` column**.

### 🔒 Leakage Prevention

* Sentiment features are **lagged (1, 2, 3 days)**
* Target variable is defined as:

```python
target = close(t + 1)
```

This ensures the model **never sees future information**.

### 📈 Rolling Features

* Rolling mean & standard deviation of prices
* Captures local trend and volatility regimes

### 🧩 Missing Data Strategy

* Sparse sentiment data is filled with **neutral values (0)**
* Rolling & lagged NaNs are imputed safely
* Rows are only dropped when the **target is missing**

---

## 🤖 Models Used

### 1️⃣ Random Forest Regressor

* Handles non-linear feature interactions
* Robust to noisy alternative data

### 2️⃣ Gradient Boosting Regressor

* Sequential error correction
* Strong performance on structured financial data

> Tree-based models were chosen for their **interpretability and reliability** over deep learning models for daily-resolution financial data.

---

## 📈 Training & Evaluation

* **Time-based train/test split**
* No random shuffling (preserves causality)
* Metrics:

  * MAE
  * RMSE

---

## 🔮 Phase-3: Mystery Stock (Live Dashboard)

A **Streamlit dashboard** is provided for Phase-3 evaluation.

### Dashboard Features

* Displays **Mystery Stock ticker**
* Fetches **live market & sentiment data**
* Predicts **next trading day closing price**
* Shows **clear timestamp** proving real-time execution
* Uses **pre-trained model for inference**

### Technology

* Streamlit
* Plotly for visualization
* Live data fetch (no cached predictions)

---

## 📁 Repository Structure

```
├── data_scrapers/
│   ├── stock_scraper.py
│   ├── news_scraper.py
│   ├── reddit_scraper.py
│   ├── trends_scraper.py
│
├── features/
│   └── feature_engineering.py
│
├── model/
│   └── predictor.py
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   └── model.pkl
│
├── main.py
├── config.py
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Train the Model (Offline)

```bash
python main.py
```

This generates:

* Trained model (`model.pkl`)
* Processed features
* Prediction logs

---

### 2️⃣ Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📝 Phase-3 Submission Artifacts

* ✅ Live Streamlit Dashboard URL
* ✅ Trained model file
* ✅ `report.txt` containing:

  * Mystery stock ticker
  * Predicted next-day price
  * Model used
  * Evaluation metrics

---

## ⚠️ Limitations & Future Work

* Sentiment data is sparse for some stocks
* No intraday modeling
* Future improvements:

  * Transformer-based sentiment
  * Regime-aware models
  * Intraday feature segmentation

---

## 🏁 Final Notes

Project StockSenseAI was designed with:

* **Correct ML principles**
* **Strict temporal discipline**
* **Explainability for evaluation**
* **Real-time deployment readiness**

The system successfully demonstrates how **market behavior + human sentiment** can be combined into a reliable predictive framework.

---


Just tell me 👍
