# StockSenseAI 📈

A full-stack, end-to-end stock prediction system that combines multi-source sentiment analysis with an ensemble machine learning model to forecast next-day stock price movements. Supports both US and Indian (NSE/BSE) markets.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Pipeline (CLI)](#running-the-pipeline-cli)
  - [Training the General Model](#training-the-general-model)
  - [Running the API Server](#running-the-api-server)
  - [Using the Frontend](#using-the-frontend)
- [API Reference](#api-reference)
- [Trading Strategies](#trading-strategies)
- [Indian Market Support](#indian-market-support)
- [Environment Variables](#environment-variables)

---

## Overview

StockSenseAI fetches historical price data, scrapes news headlines, Google Trends interest, and Reddit sentiment — engineers scale-free features from all of them — and trains an RF + GBM + XGBoost ensemble to predict next-day returns. A FastAPI backend exposes all of this as a REST API, and a single-file HTML frontend provides a Bloomberg-style screener UI.

---

## Project Structure

```
StockSenseAI/
├── api.py                        # FastAPI backend (v4)
├── main.py                       # CLI end-to-end pipeline
├── train_general_model.py        # One-time multi-stock general model training
├── config.py                     # All configuration parameters
├── requirements.txt
├── .env                          # API keys (not committed)
├── .env.example
│
├── data_scrapers/
│   ├── stock_scraper.py          # yfinance OHLCV + technical indicators
│   ├── news_scraper.py           # FinViz, NewsAPI, AlphaVantage, Moneycontrol, ET
│   ├── trends_scraper.py         # Google Trends (pytrends) with retry/fallback
│   └── reddit_scraper.py         # Pushshift + Reddit public API, multi-subreddit
│
├── features/
│   └── feature_engineering.py   # Scale-free feature pipeline, Indian circuit features
│
├── model/
│   ├── predictor.py              # Per-stock RF + GBM + XGBoost ensemble
│   └── general_predictor.py     # Multi-stock general model (same interface)
│
├── Strategy/
│   ├── backtest.py               # Event-driven backtester with stop-loss integration
│   ├── stop_loss.py              # Fixed / Trailing / ATR stop-loss manager
│   └── martingale.py             # Martingale position-sizing backtester
│
├── frontend/
│   └── screener.html             # Single-file Bloomberg-style screener UI
│
└── outputs/                      # Generated at runtime
    ├── models/
    │   ├── general_model.pkl
    │   └── <TICKER>.pkl
    ├── processed_features.csv
    ├── prediction_log.csv
    └── backtest_log.csv
```

---

## Features

**Data Sources**
- Historical OHLCV price data via yfinance
- News sentiment from FinViz, NewsAPI (3-page pagination), and AlphaVantage
- Indian news from Moneycontrol and Economic Times RSS feeds
- Google Trends interest with weekly→daily resampling and search momentum/spike features
- Reddit sentiment from 6 subreddits (wallstreetbets, stocks, investing, options, IndianStockMarket, StockMarketIndia) via Pushshift with public API fallback

**Machine Learning**
- Ensemble of Random Forest, Gradient Boosting, and XGBoost regressors (weighted 30/30/40)
- Scale-free features (ratios, not raw prices) so one model generalises across stocks at any price level
- General model trained on 20+ stocks simultaneously — available instantly for any new ticker without retraining
- Per-stock model training as a fallback (cached for subsequent requests)
- Train/test overfitting diagnostics printed at evaluation time

**Trading & Risk**
- Event-driven backtester with configurable confidence threshold
- Three stop-loss modes: Fixed %, Trailing %, ATR-based
- Hard floor override on all stop-loss modes
- Martingale position-sizing strategy backtester
- Full trade log with BUY/SELL/STOP_EXIT/FINAL_EXIT labels and per-trade P&L

**API & Frontend**
- FastAPI backend with background job queue for long-running prediction pipelines
- Batch screener endpoint for multiple tickers in parallel
- Single-ticker detail endpoint with full OHLCV history
- Ticker search endpoint
- Responsive HTML frontend: sparkline cards, Chart.js price chart, history table, overview grid, AI prediction panel with live progress bar

---

## How It Works

```
Stock Price Data  ──┐
News Sentiment    ──┤
Google Trends     ──┼──► Feature Engineering ──► Ensemble Model ──► Predicted Return
Reddit Sentiment  ──┘                                                      │
                                                                           ▼
                                                               BUY / SELL Signal
                                                                           │
                                                                           ▼
                                                               Stop-Loss Manager
                                                                           │
                                                                           ▼
                                                                    Backtester
```

**Feature engineering** produces only scale-free features (price ratios, RSI normalised to 0–1, MACD divided by close price, volume ratio vs 20-day average, etc.) so the model is not anchored to any particular price level.

**Prediction priority** in the API: (1) General model if available — instant, no training; (2) Cached per-stock model; (3) Fresh per-stock training from scratch.

---

## Installation

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/yourname/StockSenseAI.git
cd StockSenseAI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in your API keys
cp .env.example .env
```

---

## Configuration

Edit `config.py` to set:

| Parameter | Default | Description |
|---|---|---|
| `STOCK_NAME` | `"TSLA"` | Ticker for the CLI pipeline |
| `START_DATE` | `"2021-01-16"` | Historical data start |
| `END_DATE` | `"2026-02-26"` | Historical data end |
| `TEST_SIZE` | `0.2` | Fraction of data held out for evaluation |
| `GENERAL_MODEL_PATH` | `outputs/models/general_model.pkl` | Where the general model is saved |
| `TRAINING_UNIVERSE` | 20 US tickers | Stocks used to train the general model |
| `OUTPUT_DIR` | `outputs/` | Root directory for all generated files |

---

## Usage

### Running the Pipeline (CLI)

Runs the full end-to-end pipeline for the ticker set in `config.py`:

```bash
python main.py
```

This will:
1. Fetch historical stock data and calculate technical indicators
2. Scrape news, Google Trends, and Reddit sentiment in parallel
3. Engineer features and save to `outputs/processed_features.csv`
4. Train the RF + GBM + XGBoost ensemble and save to `outputs/trained_model.pkl`
5. Evaluate on the held-out test set and print overfitting diagnostics
6. Save a prediction log to `outputs/prediction_log.csv`
7. Run a backtest with trailing stop-loss and print a full performance summary

### Training the General Model

Train once on the full `TRAINING_UNIVERSE` (20 stocks by default). This model is then used for instant predictions on any ticker without retraining:

```bash
python train_general_model.py
```

Re-run monthly to refresh the model with more recent data. The script fetches all stocks in parallel (3 at a time to respect API rate limits), prints a feature importance table, and saves the model to `outputs/models/general_model.pkl`.

### Running the API Server

```bash
uvicorn api:app --reload --port 8000
```

The server loads the general model at startup. If no general model is found, it falls back to per-stock training on first request.

### Using the Frontend

Open `frontend/screener.html` directly in your browser. The page connects to the API at `http://localhost:8000` and will show a connection status indicator in the top-right corner.

The frontend supports:
- Viewing live price cards for the default watchlist (META, AAPL, AMZN, NFLX, GOOGL)
- Searching any ticker including Indian stocks (e.g. `TCS.NS`, `RELIANCE.NS`)
- Charting price history with 1W / 1M / 3M / 6M / 1Y range selectors
- Browsing OHLCV history and stock overview data
- Running the AI prediction pipeline with a live progress bar and result card

---

## API Reference

All endpoints are served at `http://localhost:8000`.

### `GET /`
Health check. Returns API status and whether the general model is loaded.

### `GET /stocks?tickers=META,AAPL,AMZN`
Batch snapshot for multiple tickers. Returns price, change, sparkline data, and key metrics for each. Fetched in parallel (up to 8 workers).

### `GET /stock/{sym}?period=1y`
Full detail for a single ticker including complete OHLCV history for the requested period (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`).

### `GET /search?q=TSLA`
Search for a ticker by symbol. Returns price, change, exchange, market cap, and currency.

### `POST /predict`
Start a prediction pipeline as a background job.

**Request body:**
```json
{
  "ticker": "TSLA",
  "start_date": "2021-01-16",
  "end_date": "2026-02-26",
  "force_retrain": false
}
```

**Response:**
```json
{ "job_id": "TSLA_143022123456" }
```

### `GET /predict/status/{job_id}`
Poll for prediction job status. Returns `status` (`queued` / `running` / `done` / `error`), `progress` (0–100), `message`, and `result` when complete.

**Result shape:**
```json
{
  "sym": "TSLA",
  "source": "general_model",
  "current_price": 250.00,
  "predicted_price": 256.25,
  "delta": 6.25,
  "delta_pct": 2.5,
  "signal": "BUY",
  "metrics": { "Test_Direction_Accuracy": 54.3, "Test_RMSE": 0.0182 },
  "timestamp": "2026-03-30T14:30:22"
}
```

---

## Trading Strategies

### Standard Backtest (`Strategy/backtest.py`)

The default backtester uses a confidence threshold on predicted returns to generate BUY / SELL signals, then delegates all risk management to the `StopLossManager`.

Exit priority (highest to lowest):
1. **Stop loss triggered** → `STOP_EXIT` (loss cut immediately, ignores ML signal)
2. **ML SELL signal** → `SELL`
3. **Last day** → `FINAL_EXIT`

Metrics reported: Strategy Return, Buy & Hold Return, Alpha, Sharpe Ratio, Max Drawdown, Win Rate, Avg Win/Loss, Profit Factor, and a breakdown of exit types.

### Stop-Loss Modes (`Strategy/stop_loss.py`)

| Mode | Description |
|---|---|
| `fixed` | Exit if price drops X% below entry |
| `trailing` | Stop tracks the peak price; exits on X% pullback from peak |
| `atr` | Stop set at N × ATR below entry; trails upward as price rises |

All modes support an optional `hard_floor_pct` that prevents losses beyond a maximum percentage regardless of mode.

### Martingale Strategy (`Strategy/martingale.py`)

After each losing trade, doubles the position size (capped at a configurable multiplier). Resets to base size on a win. Useful for comparing against the standard strategy but carries significant ruin risk in extended drawdowns.

---

## Indian Market Support

Append `.NS` (NSE) or `.BO` (BSE) to any ticker:

```
RELIANCE.NS   TCS.NS   INFY.NS   HDFCBANK.NS
```

Indian tickers automatically get:
- News from Moneycontrol and Economic Times instead of FinViz
- Indian subreddits (`r/IndianStockMarket`, `r/StockMarketIndia`) included in Reddit fetch
- Google Trends query stripped of `.NS`/`.BO` suffix with yfinance company name fallback
- Two extra model features: `upper_circuit_proximity` and `lower_circuit_proximity` (NSE/BSE impose a 10% daily circuit breaker)
- `is_indian_stock` binary flag so the general model knows the market context
- Holiday gap handling: known NSE/BSE holidays are forward-filled rather than left as NaN rows

---

## Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
NEWS_API_KEY=your_newsapi_org_key
AV_API_KEY=your_alphavantage_key
HF_TOKEN=your_huggingface_token
```

| Variable | Source | Free Tier |
|---|---|---|
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) | 100 requests/day, 30-day history |
| `AV_API_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | 25 requests/day |
| `HF_TOKEN` | [huggingface.co](https://huggingface.co/settings/tokens) | Free |

The system degrades gracefully if any key is missing — that source is skipped and the others are still used.
