"""
StockSense AI — FastAPI Backend  (v4)
======================================
Changes vs v3:
  - Scraper imports updated (no more *2 suffixes)
  - COMPANY_ALIASES removed — TrendsScraper handles yfinance fallback internally
  - ticker passed to combine_all_features() for Indian feature support
  - currency field added to /stocks and /search responses

Run from project root:
    uvicorn api:app --reload --port 8000

Train the general model first (one-time):
    python train_general_model.py
"""

import sys, os, traceback, threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import numpy as np
import pandas as pd

import config
from data_scrapers.stock_scraper   import StockScraper
from data_scrapers.news_scraper    import NewsScraper
from data_scrapers.trends_scraper  import TrendsScraper
from data_scrapers.reddit_scraper  import RedditScraper
from features.feature_engineering  import FeatureEngineer
from model.predictor               import StockPredictor
from model.general_predictor       import GeneralPredictor

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="StockSense AI", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP — load general model once, keep in memory for all requests
# ══════════════════════════════════════════════════════════════════════════════
GENERAL_MODEL_PATH = getattr(config, 'GENERAL_MODEL_PATH',
                              'outputs/models/general_model.pkl')
MODEL_DIR   = "outputs/models"
FEATURE_DIR = "outputs/features"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

_general_model: GeneralPredictor | None = None


@app.on_event("startup")
def load_general_model():
    global _general_model
    gp = GeneralPredictor()
    if gp.is_available(GENERAL_MODEL_PATH):
        gp.load_model(GENERAL_MODEL_PATH)
        _general_model = gp
        print(f"✓ GeneralPredictor loaded from {GENERAL_MODEL_PATH}")
    else:
        print(f"⚠  No general model found at {GENERAL_MODEL_PATH}.")
        print(f"   Run:  python train_general_model.py   to train it.")
        _general_model = None


# ── Job store ─────────────────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock  = threading.Lock()


def _set_job(job_id: str, **kw):
    with _jobs_lock:
        _jobs[job_id].update(kw)


def _model_path(sym):   return f"{MODEL_DIR}/{sym}.pkl"
def _feature_path(sym): return f"{FEATURE_DIR}/{sym}.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    return {
        "status":        "ok",
        "service":       "StockSense AI API v4",
        "general_model": _general_model is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA — batch light snapshot (screener cards)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/stocks")
def get_stocks(tickers: str = "META,AAPL,AMZN,NFLX,GOOGL"):
    syms    = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    results = []

    def _fetch_one(sym):
        t    = yf.Ticker(sym)
        info = t.info or {}

        hist = t.history(period="6mo", interval="1d")
        hist.reset_index(inplace=True)
        closes = [round(float(r["Close"]), 2)
                  for _, r in hist.iterrows() if not np.isnan(r["Close"])]
        dates  = [str(r["Date"])[:10] for _, r in hist.iterrows()]

        price      = float(info.get("currentPrice") or
                           info.get("regularMarketPrice") or
                           (closes[-1] if closes else 0))
        prev_close = float(info.get("previousClose") or
                           (closes[-2] if len(closes) > 1 else price))
        chg        = round(price - prev_close, 2)
        chg_pct    = round((chg / prev_close * 100) if prev_close else 0, 3)

        return {
            "sym":         sym,
            "name":        info.get("longName") or info.get("shortName") or sym,
            "sector":      info.get("sector", ""),
            "price":       round(price, 2),
            "prev_close":  round(prev_close, 2),
            "chg":         chg,
            "chg_pct":     chg_pct,
            "open":        round(float(info.get("open") or
                                      info.get("regularMarketOpen") or 0), 2),
            "day_high":    round(float(info.get("dayHigh")  or 0), 2),
            "day_low":     round(float(info.get("dayLow")   or 0), 2),
            "volume":      info.get("volume") or 0,
            "mkt_cap":     info.get("marketCap") or 0,
            "pe":          round(float(info.get("trailingPE") or 0), 2),
            "week52_high": round(float(info.get("fiftyTwoWeekHigh") or 0), 2),
            "week52_low":  round(float(info.get("fiftyTwoWeekLow")  or 0), 2),
            "avg_volume":  info.get("averageVolume") or 0,
            "beta":        round(float(info.get("beta") or 0), 2),
            "currency":    info.get("currency", "USD"),
            "sparkline":   list(zip(dates[-90:], closes[-90:])),
        }

    with ThreadPoolExecutor(max_workers=min(len(syms), 8)) as ex:
        futs = {ex.submit(_fetch_one, s): s for s in syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"sym": sym, "error": True, "detail": str(e)})

    order = {s: i for i, s in enumerate(syms)}
    results.sort(key=lambda x: order.get(x.get("sym", ""), 999))
    return {"stocks": results, "fetched_at": datetime.now().isoformat()}


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA — single ticker full detail + history
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/stock/{sym}")
def get_stock(sym: str, period: str = "1y"):
    sym = sym.upper().strip()
    try:
        t    = yf.Ticker(sym)
        info = t.info or {}

        hist = t.history(period=period, interval="1d")
        if hist.empty:
            raise HTTPException(404, f"No data found for {sym}")
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].dt.tz_localize(None)

        history = [
            {
                "date":   row["Date"].strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
            for _, row in hist.iterrows()
        ]

        price      = float(info.get("currentPrice") or
                           info.get("regularMarketPrice") or
                           history[-1]["close"])
        prev_close = float(info.get("previousClose") or
                           (history[-2]["close"] if len(history) > 1 else price))
        chg        = round(price - prev_close, 2)
        chg_pct    = round((chg / prev_close * 100) if prev_close else 0, 3)

        return {
            "sym":         sym,
            "name":        info.get("longName") or info.get("shortName") or sym,
            "sector":      info.get("sector", ""),
            "industry":    info.get("industry", ""),
            "country":     info.get("country", ""),
            "currency":    info.get("currency", "USD"),
            "exchange":    info.get("exchange", ""),
            "price":       round(price, 2),
            "prev_close":  round(prev_close, 2),
            "chg":         chg,
            "chg_pct":     chg_pct,
            "open":        round(float(info.get("open") or
                                      info.get("regularMarketOpen") or
                                      history[-1]["open"]), 2),
            "day_high":    round(float(info.get("dayHigh")  or history[-1]["high"]), 2),
            "day_low":     round(float(info.get("dayLow")   or history[-1]["low"]),  2),
            "week52_high": round(float(info.get("fiftyTwoWeekHigh") or 0), 2),
            "week52_low":  round(float(info.get("fiftyTwoWeekLow")  or 0), 2),
            "volume":      info.get("volume") or info.get("regularMarketVolume") or 0,
            "avg_volume":  info.get("averageVolume") or 0,
            "mkt_cap":     info.get("marketCap") or 0,
            "pe":          round(float(info.get("trailingPE") or 0), 2),
            "eps":         round(float(info.get("trailingEps") or 0), 2),
            "beta":        round(float(info.get("beta") or 0), 2),
            "div_yield":   round(float((info.get("dividendYield") or 0) * 100), 2),
            "target_mean": round(float(info.get("targetMeanPrice") or 0), 2),
            "history":     history,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/search")
def search_ticker(q: str):
    q = q.strip().upper()
    results = []
    try:
        t     = yf.Ticker(q)
        info  = t.info or {}
        price = float(info.get("currentPrice") or
                      info.get("regularMarketPrice") or 0)
        if price > 0:
            prev = float(info.get("previousClose") or price)
            results.append({
                "sym":      q,
                "name":     info.get("longName") or info.get("shortName") or q,
                "sector":   info.get("sector", ""),
                "price":    round(price, 2),
                "chg_pct":  round((price - prev) / prev * 100 if prev else 0, 3),
                "exchange": info.get("exchange", ""),
                "mkt_cap":  info.get("marketCap") or 0,
                "currency": info.get("currency", "USD"),
            })
    except Exception:
        pass
    return {"results": results, "query": q}


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    ticker:        str
    start_date:    str  = config.START_DATE
    end_date:      str  = config.END_DATE
    force_retrain: bool = False


@app.post("/predict")
def start_predict(req: PredictRequest, bg: BackgroundTasks):
    sym    = req.ticker.upper().strip()
    job_id = f"{sym}_{datetime.now().strftime('%H%M%S%f')}"
    with _jobs_lock:
        _jobs[job_id] = {
            "status":   "queued",
            "progress": 0,
            "message":  "Queued",
            "result":   None,
            "error":    None,
        }
    bg.add_task(_run_pipeline, job_id, sym,
                req.start_date, req.end_date, req.force_retrain)
    return {"job_id": job_id}


@app.get("/predict/status/{job_id}")
def predict_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL SENTIMENT FETCH
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_sentiment_parallel(sym: str, start_date: str, end_date: str) -> dict:
    """
    Fetch news, Google Trends, Reddit IN PARALLEL.
    - NewsScraper.get_daily_sentiment() already routes Indian vs US tickers
    - TrendsScraper.get_search_trends() already strips .NS/.BO and falls
      back to yfinance company name internally — no COMPANY_ALIASES needed
    - RedditScraper.get_daily_reddit_sentiment() already adds Indian
      subreddits for .NS/.BO tickers
    """
    def fetch_news():
        return NewsScraper().get_daily_sentiment(sym)

    def fetch_trends():
        return TrendsScraper().get_search_trends(sym, start_date, end_date)

    def fetch_reddit():
        return RedditScraper().get_daily_reddit_sentiment(sym)

    result = {}
    tasks  = {'news': fetch_news, 'trends': fetch_trends, 'reddit': fetch_reddit}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                result[name] = fut.result()
            except Exception as e:
                print(f"  ⚠  {name} fetch failed for {sym}: {e}")
                result[name] = pd.DataFrame()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE  (background thread)
# ══════════════════════════════════════════════════════════════════════════════
def _run_pipeline(job_id: str, sym: str,
                  start_date: str, end_date: str, force_retrain: bool):
    try:
        engineer = FeatureEngineer()

        # ── STEP 1 — fetch stock data ──────────────────────────────────────
        _set_job(job_id, status="running", progress=5,
                 message="Fetching historical stock data…")
        sc       = StockScraper(sym)
        stock_df = sc.fetch_historical_data(start_date, end_date)
        if stock_df.empty:
            raise ValueError(f"No stock data returned for {sym}")
        stock_df = sc.calculate_technical_indicators(stock_df)

        # ── STEP 2 — parallel sentiment ────────────────────────────────────
        _set_job(job_id, progress=20,
                 message="Fetching news, Google Trends & Reddit in parallel…")
        sentiment = _fetch_sentiment_parallel(sym, start_date, end_date)

        # ── STEP 3 — feature engineering ──────────────────────────────────
        _set_job(job_id, progress=45, message="Engineering features…")
        combined_df = engineer.combine_all_features(
            stock_df,
            sentiment.get('news',   pd.DataFrame()),
            sentiment.get('trends', pd.DataFrame()),
            sentiment.get('reddit', pd.DataFrame()),
            ticker=sym,                          # enables Indian features
        )

        # ── PRIORITY 1: General model (instant — no training) ──────────────
        if _general_model is not None and not force_retrain:
            _set_job(job_id, progress=70,
                     message="Running GeneralPredictor (no training needed)…")

            feature_cols = _general_model.feature_columns

            # Align combined_df to the general model's feature set
            for col in feature_cols:
                if col not in combined_df.columns:
                    combined_df[col] = 0.0
            engineer.feature_columns = feature_cols

            combined_df.to_csv(_feature_path(sym), index=False)

            split_idx = int(len(combined_df) * (1 - config.TEST_SIZE))
            test_df   = combined_df.iloc[split_idx:].copy().reset_index(drop=True)
            X_train   = combined_df.iloc[:split_idx][feature_cols]
            y_train   = combined_df.iloc[:split_idx]['target']
            X_test    = test_df[feature_cols]
            y_test    = test_df['target']

            _general_model.X_train = X_train
            _general_model.y_train = y_train
            metrics, _ = _general_model.evaluate(X_test, y_test)

            latest_return   = _general_model.predict(
                combined_df[feature_cols].iloc[-1:])[0]
            current_price   = float(combined_df['close'].iloc[-1])
            predicted_price = round(current_price * (1 + latest_return), 2)

            _set_job(job_id, status="done", progress=100,
                     message="Done — GeneralPredictor",
                     result=_build_result(sym, current_price, predicted_price,
                                          metrics, "general_model"))
            return

        # ── PRIORITY 2: Cached per-stock model ────────────────────────────
        mp = _model_path(sym)
        fp = _feature_path(sym)
        predictor = StockPredictor()

        if os.path.exists(mp) and os.path.exists(fp) and not force_retrain:
            _set_job(job_id, progress=70,
                     message="Loading cached per-stock model…")
            predictor.load_model(mp)

            cached_df = pd.read_csv(fp)
            cached_df["Date"] = pd.to_datetime(cached_df["Date"])
            exclude = ('Date', 'target', 'date', 'close', 'open',
                       'high', 'low', 'volume')
            engineer.feature_columns = [
                c for c in cached_df.columns if c not in exclude]

            _, X_test, _, y_test, test_df = engineer.prepare_train_test_split(
                cached_df, test_size=config.TEST_SIZE)
            metrics, _ = predictor.evaluate(X_test, y_test)

            latest_return   = predictor.predict(
                cached_df[engineer.feature_columns].iloc[-1:])[0]
            current_price   = float(cached_df['close'].iloc[-1])
            predicted_price = round(current_price * (1 + latest_return), 2)

            _set_job(job_id, status="done", progress=100,
                     message="Done — cached per-stock model",
                     result=_build_result(sym, current_price, predicted_price,
                                          metrics, "cached"))
            return

        # ── PRIORITY 3: Train per-stock model from scratch ─────────────────
        _set_job(job_id, progress=60,
                 message="Training RF + GBM + XGBoost ensemble…")
        combined_df.to_csv(fp, index=False)

        X_train, X_test, y_train, y_test, test_df = \
            engineer.prepare_train_test_split(
                combined_df, test_size=config.TEST_SIZE)
        predictor.train(X_train, y_train)
        predictor.save_model(mp)

        _set_job(job_id, progress=88, message="Evaluating model…")
        metrics, _ = predictor.evaluate(X_test, y_test)

        latest_return   = predictor.predict(
            combined_df[engineer.feature_columns].iloc[-1:])[0]
        current_price   = float(stock_df['close'].iloc[-1])
        predicted_price = round(current_price * (1 + latest_return), 2)

        _set_job(job_id, status="done", progress=100,
                 message="Done — freshly trained per-stock model",
                 result=_build_result(sym, current_price, predicted_price,
                                      metrics, "trained"))

    except Exception:
        _set_job(job_id, status="error", progress=100,
                 message="Pipeline failed",
                 error=traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _build_result(sym, cur, pred, metrics, source):
    delta     = round(pred - cur, 2)
    delta_pct = round((delta / cur * 100) if cur else 0, 3)
    return {
        "sym":             sym,
        "source":          source,
        "current_price":   round(cur, 2),
        "predicted_price": pred,
        "delta":           delta,
        "delta_pct":       delta_pct,
        "signal":          "BUY" if delta > 0 else "SELL",
        "metrics":         {k: round(float(v), 4)
                            for k, v in (metrics or {}).items()},
        "timestamp":       datetime.now().isoformat(),
    }