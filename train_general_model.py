# train_general_model.py
"""
One-time training script for the StockSenseAI General Model.

Run once from your project root:
    python train_general_model.py

What it does:
  1. Fetches full historical data + all sentiment for every stock
     in config.TRAINING_UNIVERSE (parallel, 3 stocks at a time)
  2. Engineers scale-free features for each stock
     (Indian tickers get circuit-breaker proximity features too)
  3. Concatenates all stock data into one giant training DataFrame
  4. Trains the RF + GBM + XGBoost ensemble on the combined data
  5. Saves the model to config.GENERAL_MODEL_PATH

Re-run this monthly to refresh the model with more recent data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import config
from data_scrapers.stock_scraper   import StockScraper
from data_scrapers.news_scraper    import NewsScraper
from data_scrapers.trends_scraper  import TrendsScraper
from data_scrapers.reddit_scraper  import RedditScraper
from features.feature_engineering  import FeatureEngineer
from model.general_predictor       import GeneralPredictor


# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_UNIVERSE  = getattr(config, 'TRAINING_UNIVERSE', [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'JPM', 'BAC', 'V', 'WMT', 'KO',
    'NFLX', 'AMD', 'UBER', 'SHOP', 'SQ',
    'DIS', 'PFE', 'XOM',
])
START_DATE         = getattr(config, 'START_DATE', '2021-01-01')
END_DATE           = getattr(config, 'END_DATE',
                             datetime.today().strftime('%Y-%m-%d'))
GENERAL_MODEL_PATH = getattr(config, 'GENERAL_MODEL_PATH',
                             'outputs/models/general_model.pkl')
TEST_SIZE          = getattr(config, 'TEST_SIZE', 0.2)

# 3 stocks in parallel — safe for Reddit/NewsAPI rate limits
PARALLEL_FETCH = 3


# ── Per-stock data fetch ───────────────────────────────────────────────────────
def fetch_stock_data(ticker):
    """
    Fetch and engineer features for a single stock.
    Returns (ticker, combined_df, feature_columns) or (ticker, None, []).
    """
    print(f"\n[{ticker}] Fetching data...")
    try:
        scraper  = StockScraper(ticker)
        stock_df = scraper.fetch_historical_data(START_DATE, END_DATE)
        if stock_df.empty:
            print(f"[{ticker}] ✗ No price data")
            return ticker, None, []
        stock_df = scraper.calculate_technical_indicators(stock_df)

        # Fetch sentiment sources in parallel
        # TrendsScraper handles .NS/.BO stripping + yfinance fallback internally
        # NewsScraper routes Indian vs US tickers to correct sources
        # RedditScraper adds Indian subreddits for .NS/.BO tickers
        def _news():
            return NewsScraper().get_daily_sentiment(ticker)

        def _trends():
            return TrendsScraper().get_search_trends(
                ticker, START_DATE, END_DATE)

        def _reddit():
            return RedditScraper().get_daily_reddit_sentiment(ticker)

        sentiment = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(fn): name
                    for name, fn in [('news', _news),
                                     ('trends', _trends),
                                     ('reddit', _reddit)]}
            for f in as_completed(futs):
                name = futs[f]
                try:
                    sentiment[name] = f.result()
                except Exception as e:
                    print(f"[{ticker}] ✗ {name} error: {e}")
                    sentiment[name] = pd.DataFrame()

        # Feature engineering — pass ticker so Indian features are computed
        eng         = FeatureEngineer()
        combined_df = eng.combine_all_features(
            stock_df,
            sentiment.get('news',   pd.DataFrame()),
            sentiment.get('trends', pd.DataFrame()),
            sentiment.get('reddit', pd.DataFrame()),
            ticker=ticker,
        )

        if combined_df.empty or len(combined_df) < 50:
            print(f"[{ticker}] ✗ Too few rows ({len(combined_df)})")
            return ticker, None, []

        # Tag rows with source ticker for diagnostics (not a model feature)
        combined_df['_ticker'] = ticker

        print(f"[{ticker}] ✓ {len(combined_df)} rows, "
              f"{len(eng.feature_columns)} features")
        return ticker, combined_df, eng.feature_columns

    except Exception as e:
        import traceback
        print(f"[{ticker}] ✗ Failed: {e}")
        traceback.print_exc()
        return ticker, None, []


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "🌍" * 30)
    print("GENERAL MODEL TRAINING — StockSenseAI")
    print(f"Universe: {', '.join(TRAINING_UNIVERSE)}")
    print(f"Timeline: {START_DATE} → {END_DATE}")
    print(f"Stocks:   {len(TRAINING_UNIVERSE)}")
    print("=" * 60)

    os.makedirs(os.path.dirname(GENERAL_MODEL_PATH), exist_ok=True)

    # ── 1. Fetch all stocks (PARALLEL_FETCH at a time) ────────────────────
    all_dfs         = []
    feature_columns = None
    failed          = []

    with ThreadPoolExecutor(max_workers=PARALLEL_FETCH) as executor:
        futures = {executor.submit(fetch_stock_data, t): t
                   for t in TRAINING_UNIVERSE}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result[1] is not None:
                    _, df, cols = result
                    all_dfs.append(df)
                    if feature_columns is None:
                        feature_columns = cols
                else:
                    failed.append(ticker)
            except Exception as e:
                print(f"[{ticker}] ✗ Unexpected error: {e}")
                failed.append(ticker)

    if not all_dfs:
        print("\n❌ No data retrieved for any stock. Aborting.")
        return False

    successful = [t for t in TRAINING_UNIVERSE if t not in failed]
    print(f"\n✓ Data collected: {len(successful)} succeeded, "
          f"{len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # ── 2. Combine all DataFrames ─────────────────────────────────────────
    print("\n⚙️  Combining datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"   Total rows: {len(combined):,}")
    print(f"   Features:   {len(feature_columns)}")

    # ── 3. Global time-based train/test split ─────────────────────────────
    combined  = combined.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(combined) * (1 - TEST_SIZE))

    train_df = combined.iloc[:split_idx]
    test_df  = combined.iloc[split_idx:]

    X_train = train_df[feature_columns]
    y_train = train_df['target']
    X_test  = test_df[feature_columns]
    y_test  = test_df['target']

    # ── 4. NaN audit + fill ───────────────────────────────────────────────
    # When many stocks are concatenated, some columns may be missing for
    # certain stocks (e.g. Indian circuit features are 0 for US stocks).
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        print(f"\n⚠️  NaN in {len(nan_cols)} columns — filling with 0")
        for col in nan_cols[:10]:
            pct = X_train[col].isna().mean() * 100
            print(f"   {col:<45} {pct:.1f}% NaN")
        if len(nan_cols) > 10:
            print(f"   ... and {len(nan_cols) - 10} more")

    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    assert not X_train.isna().any().any(), "NaN still in X_train!"
    assert not X_test.isna().any().any(),  "NaN still in X_test!"

    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # ── 5. Train ──────────────────────────────────────────────────────────
    gp = GeneralPredictor()
    gp.train(X_train, y_train, universe=successful)

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    metrics, _ = gp.evaluate(X_test, y_test)

    # ── 7. Feature importance ─────────────────────────────────────────────
    print("\n📊 TOP 15 FEATURES (by Random Forest importance):")
    importance = gp.get_feature_importance(top_n=15)
    for feat, imp in importance.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<40} {imp:.4f}  {bar}")

    # ── 8. Save ───────────────────────────────────────────────────────────
    gp.save_model(GENERAL_MODEL_PATH)

    feat_col_path = GENERAL_MODEL_PATH.replace('.pkl', '_feature_columns.txt')
    with open(feat_col_path, 'w') as f:
        f.write('\n'.join(feature_columns))
    print(f"✓ Feature column list saved → {feat_col_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ GENERAL MODEL TRAINING COMPLETE")
    print(f"   Stocks:     {len(successful)}")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Train DA:   {metrics['Train_DA']:.1f}%")
    print(f"   Test  DA:   {metrics['Test_DA']:.1f}%")
    print(f"   DA Gap:     {metrics['Test_DA'] - metrics['Train_DA']:+.1f}%")
    print(f"   Saved to:   {GENERAL_MODEL_PATH}")
    print("=" * 60)
    print("\nRe-run this script monthly to keep the model fresh.\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)