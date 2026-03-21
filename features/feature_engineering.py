# features/feature_engineering.py
"""
Feature engineering pipeline — generalisation-ready.

KEY CHANGE vs original:
  All price-based features are now RATIOS (scale-free).
  A model trained on AAPL at $180 can now predict TSLA at $250
  because no feature contains a raw dollar value — only relationships
  between prices. This is the prerequisite for a general model.

  Raw dollar columns (open, high, low, close, volume) are kept in the
  DataFrame for the backtester and prediction display but are EXCLUDED
  from self.feature_columns so the model never sees them.

Ratio features replacing raw price features:
  price_vs_sma5        close / SMA5          (above short MA?)
  price_vs_sma20       close / SMA20         (above long MA?)
  sma5_vs_sma20        SMA5 / SMA20          (golden/death cross)
  high_low_range       (high-low) / close    (daily range %)
  close_vs_open        (close-open) / open   (intraday direction)
  macd_normalised      MACD / close          (MACD scaled to price)
  macd_signal_gap      (MACD-signal) / close (crossover strength)
  volume_ratio         volume / 20d avg      (abnormal volume)
  volatility_pct       rolling std / close   (volatility %)
  price_momentum_Nd    N-day pct return      (already scale-free)
  rsi_norm             RSI / 100             (normalised to 0-1)
  price_vs_smaW        close / rolling(W)    (rolling ratio)
  volatility_ratio_W   std(W) / close        (rolling vol ratio)
"""
import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    # ------------------------------------------------------------------ #
    #  Lagged features                                                     #
    # ------------------------------------------------------------------ #
    def create_lagged_features(self, df, columns, lags=[1, 3, 7]):
        df = df.copy()
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    # ------------------------------------------------------------------ #
    #  Rolling ratio features (replaces raw rolling close)                #
    # ------------------------------------------------------------------ #
    def create_rolling_features(self, df, windows=[3, 7, 14]):
        df = df.copy()
        for w in windows:
            roll_mean = df['close'].rolling(window=w).mean()
            roll_std  = df['close'].rolling(window=w).std()
            df[f'price_vs_sma{w}']      = df['close'] / (roll_mean + 1e-9)
            df[f'volatility_ratio_{w}'] = roll_std    / (df['close'] + 1e-9)
        return df

    # ------------------------------------------------------------------ #
    #  Main combiner                                                       #
    # ------------------------------------------------------------------ #
    def combine_all_features(self, stock_df, news_df, trends_df, reddit_df):

        df = stock_df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Normalise all column names to lowercase so stock_scraper.py column
        # naming (SMA_5, SMA_20, RSI etc.) doesn't cause KeyErrors here.
        df.columns = df.columns.str.lower()
        # Restore the Date column with capital D — merge logic depends on it
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)

        # ── Merge sentiment sources ───────────────────────────────────
        if news_df is not None and not news_df.empty:
            news_df = news_df.copy()
            news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
            df = df.merge(news_df, on='Date', how='left')
        else:
            for c in ['avg_sentiment', 'sentiment_std',
                      'positive_ratio', 'negative_ratio', 'news_volume']:
                df[c] = 0.0

        if reddit_df is not None and not reddit_df.empty:
            reddit_df = reddit_df.copy()
            reddit_df['Date'] = pd.to_datetime(reddit_df['Date']).dt.tz_localize(None)
            df = df.merge(reddit_df, on='Date', how='left')
        else:
            for c in ['reddit_avg_sentiment', 'reddit_weighted_sentiment',
                      'reddit_volume', 'reddit_engagement',
                      'reddit_positive_ratio']:
                df[c] = 0.0

        if trends_df is not None and not trends_df.empty:
            trends_df = trends_df.copy()
            trends_df['Date'] = pd.to_datetime(trends_df['Date']).dt.tz_localize(None)
            trend_cols = ['Date', 'search_interest']
            for extra in ['search_momentum', 'search_spike']:
                if extra in trends_df.columns:
                    trend_cols.append(extra)
            df = df.merge(trends_df[trend_cols], on='Date', how='left')
            df['search_interest'] = df['search_interest'].ffill().fillna(0)
        else:
            df['search_interest'] = 0.0

        # ── Fill and clip sentiment ───────────────────────────────────
        sentiment_cols = [
            c for c in df.columns if any(x in c for x in
            ['sentiment', 'reddit', 'news', 'ratio',
             'volume', 'engagement', 'search', 'controversy'])
        ]
        df[sentiment_cols] = df[sentiment_cols].fillna(0)

        clip_cols = [c for c in df.columns if any(
            x in c for x in ['sentiment', 'ratio', 'controversy']
        )]
        df[clip_cols] = df[clip_cols].clip(-1, 1)

        # ── Scale-free price & technical features ────────────────────
        eps = 1e-9

        df['price_vs_sma5']   = df['close'] / (df['sma_5']  + eps)
        df['price_vs_sma20']  = df['close'] / (df['sma_20'] + eps)
        df['sma5_vs_sma20']   = df['sma_5']  / (df['sma_20'] + eps)
        df['high_low_range']  = (df['high'] - df['low'])   / (df['close'] + eps)
        df['close_vs_open']   = (df['close'] - df['open']) / (df['open']  + eps)
        df['macd_normalised'] = df['macd'] / (df['close'] + eps)
        df['macd_signal_gap'] = (df['macd'] - df['signal_line']) / (df['close'] + eps)
        df['volatility_pct']  = df['volatility'] / (df['close'] + eps)
        df['rsi_norm']        = df['rsi'] / 100.0

        vol_avg = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (vol_avg + eps)

        for d in [1, 3, 5, 10]:
            df[f'price_momentum_{d}d'] = df['close'].pct_change(d)

        df = self.create_rolling_features(df, windows=[3, 7, 14])
        df = self.create_lagged_features(df, sentiment_cols, lags=[1, 2, 3])

        # ── Target ───────────────────────────────────────────────────
        df['target'] = df['close'].pct_change(-1)
        df = df.dropna(subset=['target'])

        # ── Fill remaining NaNs ───────────────────────────────────────
        scale_free_tech = [
            'price_vs_sma5', 'price_vs_sma20', 'sma5_vs_sma20',
            'high_low_range', 'close_vs_open', 'macd_normalised',
            'macd_signal_gap', 'volume_ratio', 'volatility_pct', 'rsi_norm',
        ] + [f'price_momentum_{d}d' for d in [1, 3, 5, 10]] \
          + [f'price_vs_sma{w}'      for w in [3, 7, 14]] \
          + [f'volatility_ratio_{w}' for w in [3, 7, 14]]

        existing = [c for c in scale_free_tech if c in df.columns]
        df[existing] = df[existing].ffill().bfill()
        df = df.fillna(0)

        # ── Feature columns — NO raw dollar values ────────────────────
        exclude = {
            'Date', 'target',
            'close', 'open', 'high', 'low', 'volume',
            'sma_5', 'sma_20', 'rsi', 'macd', 'signal_line',
            'volatility', 'price_change',
        }
        self.feature_columns = [c for c in df.columns if c not in exclude]
        return df

    # ------------------------------------------------------------------ #
    #  Train / test split                                                  #
    # ------------------------------------------------------------------ #
    def prepare_train_test_split(self, df, test_size=0.2):
        split_idx = int(len(df) * (1 - test_size))
        train_df  = df.iloc[:split_idx]
        test_df   = df.iloc[split_idx:]
        X_train   = train_df[self.feature_columns]
        y_train   = train_df['target']
        X_test    = test_df[self.feature_columns]
        y_test    = test_df['target']
        return X_train, X_test, y_train, y_test, test_df