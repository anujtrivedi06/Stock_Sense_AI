# features/feature_engineering.py
"""
Feature engineering pipeline — generalisation-ready, supports both
US and Indian (NSE/BSE) stocks.

All price-based features are RATIOS (scale-free) so a model trained on
AAPL at $180 can predict TCS.NS at Rs3500 without raw price leakage.

Indian-specific additions (when ticker ends in .NS or .BO):
  upper_circuit_proximity — how close price is to the 10% upper circuit limit
  lower_circuit_proximity — how close price is to the 10% lower circuit limit
  is_indian_stock         — binary flag so the model knows the market context
  Holiday gaps from INDIAN_MARKET_HOLIDAYS in config are forward-filled
  rather than left as NaN rows.

New parameter:
  ticker (str) — pass the ticker symbol to enable Indian features.
                 Defaults to "" so existing callers without the param still work.
"""
import pandas as pd
import numpy as np
import config


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
    #  Rolling ratio features                                              #
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
    def combine_all_features(self, stock_df, news_df, trends_df,
                             reddit_df, ticker=""):
        """
        Parameters
        ----------
        stock_df   : DataFrame from StockScraper
        news_df    : DataFrame from NewsScraper
        trends_df  : DataFrame from TrendsScraper
        reddit_df  : DataFrame from RedditScraper
        ticker     : str — ticker symbol, used to enable Indian-specific
                     features. Defaults to "" (no Indian features).
        """
        is_indian = (ticker.upper().endswith('.NS') or
                     ticker.upper().endswith('.BO'))

        df = stock_df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Normalise column names to lowercase
        df.columns = df.columns.str.lower()
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)

        # ── Drop known Indian market holiday rows where price is NaN ─────
        # NSE/BSE have more holidays than NYSE. When price data has no
        # entry for a holiday but sentiment data does, we get NaN price
        # rows after the merge. We remove them here before that happens.
        if is_indian:
            indian_holidays = getattr(config, 'INDIAN_MARKET_HOLIDAYS', [])
            holiday_dates   = pd.to_datetime(indian_holidays)
            df = df[~(
                df['Date'].isin(holiday_dates) & df['close'].isna()
            )]

        # ── Merge sentiment sources ───────────────────────────────────────
        if news_df is not None and not news_df.empty:
            news_df = news_df.copy()
            news_df['Date'] = pd.to_datetime(
                news_df['Date']).dt.tz_localize(None)
            df = df.merge(news_df, on='Date', how='left')
        else:
            for c in ['avg_sentiment', 'sentiment_std',
                      'positive_ratio', 'negative_ratio', 'news_volume']:
                df[c] = 0.0

        if reddit_df is not None and not reddit_df.empty:
            reddit_df = reddit_df.copy()
            reddit_df['Date'] = pd.to_datetime(
                reddit_df['Date']).dt.tz_localize(None)
            df = df.merge(reddit_df, on='Date', how='left')
        else:
            for c in ['reddit_avg_sentiment', 'reddit_weighted_sentiment',
                      'reddit_volume', 'reddit_engagement',
                      'reddit_positive_ratio']:
                df[c] = 0.0

        if trends_df is not None and not trends_df.empty:
            trends_df = trends_df.copy()
            trends_df['Date'] = pd.to_datetime(
                trends_df['Date']).dt.tz_localize(None)
            trend_cols = ['Date', 'search_interest']
            for extra in ['search_momentum', 'search_spike']:
                if extra in trends_df.columns:
                    trend_cols.append(extra)
            df = df.merge(trends_df[trend_cols], on='Date', how='left')
            df['search_interest'] = df['search_interest'].ffill().fillna(0)
        else:
            df['search_interest'] = 0.0

        # ── Fill and clip sentiment columns ───────────────────────────────
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

        # ── Scale-free price & technical features ─────────────────────────
        eps = 1e-9

        df['price_vs_sma5']   = df['close'] / (df['sma_5']  + eps)
        df['price_vs_sma20']  = df['close'] / (df['sma_20'] + eps)
        df['sma5_vs_sma20']   = df['sma_5']  / (df['sma_20'] + eps)
        df['high_low_range']  = (df['high'] - df['low'])   / (df['close'] + eps)
        df['close_vs_open']   = (df['close'] - df['open']) / (df['open']  + eps)
        df['macd_normalised'] = df['macd'] / (df['close'] + eps)
        df['macd_signal_gap'] = (
            (df['macd'] - df['signal_line']) / (df['close'] + eps)
        )
        df['volatility_pct']  = df['volatility'] / (df['close'] + eps)
        df['rsi_norm']        = df['rsi'] / 100.0

        vol_avg = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (vol_avg + eps)

        for d in [1, 3, 5, 10]:
            df[f'price_momentum_{d}d'] = df['close'].pct_change(d)

        df = self.create_rolling_features(df, windows=[3, 7, 14])
        df = self.create_lagged_features(df, sentiment_cols, lags=[1, 2, 3])

        # ── Indian-specific features ──────────────────────────────────────
        # NSE/BSE impose a 10% daily circuit breaker. Being close to the
        # upper or lower circuit is a meaningful signal (momentum / reversal).
        # For US stocks both features are 0 so the model ignores them.
        df['is_indian_stock'] = 1.0 if is_indian else 0.0

        if is_indian:
            circuit_range = df['close'] * 0.10   # 10% of close price
            df['upper_circuit_proximity'] = (
                (df['high'] - df['close']) / (circuit_range + eps)
            ).clip(0, 1)
            df['lower_circuit_proximity'] = (
                (df['close'] - df['low']) / (circuit_range + eps)
            ).clip(0, 1)
        else:
            df['upper_circuit_proximity'] = 0.0
            df['lower_circuit_proximity'] = 0.0

        # ── Target ────────────────────────────────────────────────────────
        df['target'] = df['close'].pct_change(-1)
        df = df.dropna(subset=['target'])

        # ── Fill remaining NaNs ───────────────────────────────────────────
        scale_free_tech = [
            'price_vs_sma5', 'price_vs_sma20', 'sma5_vs_sma20',
            'high_low_range', 'close_vs_open', 'macd_normalised',
            'macd_signal_gap', 'volume_ratio', 'volatility_pct', 'rsi_norm',
            'is_indian_stock',
            'upper_circuit_proximity', 'lower_circuit_proximity',
        ] + [f'price_momentum_{d}d' for d in [1, 3, 5, 10]] \
          + [f'price_vs_sma{w}'      for w in [3, 7, 14]] \
          + [f'volatility_ratio_{w}' for w in [3, 7, 14]]

        existing = [c for c in scale_free_tech if c in df.columns]
        df[existing] = df[existing].ffill().bfill()
        df = df.fillna(0)

        # ── Feature columns — NO raw price / volume values ────────────────
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