# features/feature_engineering.py
"""
Feature engineering pipeline
CRITICAL: Prevents temporal leakage by only using past data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
    
    def create_lagged_features(self, df, columns, lags=[1, 3, 7]):
        """
        Create lagged features to prevent temporal leakage
        """
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, column, windows=[3, 7, 14]):
        """
        Create rolling mean and std features
        """
        df = df.copy()
        
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
        
        return df
    
    def combine_all_features(self, stock_df, news_df, trends_df, reddit_df):
        """
        Combine all features into a single DataFrame
        Ensures STRICT temporal alignment and prevents leakage
        """

        df = stock_df.copy()

        # ---- Normalize Date column everywhere ----
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # ---- Merge NEWS sentiment (daily) ----
        if news_df is not None and not news_df.empty:
            news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
            df = df.merge(news_df, on='Date', how='left')
        else:
            df[['avg_sentiment', 'sentiment_std',
                'positive_ratio', 'negative_ratio', 'news_volume']] = 0

        # ---- Merge REDDIT sentiment (daily) ----
        if reddit_df is not None and not reddit_df.empty:
            reddit_df['Date'] = pd.to_datetime(reddit_df['Date']).dt.tz_localize(None)
            df = df.merge(reddit_df, on='Date', how='left')
        else:
            df[['reddit_avg_sentiment', 'reddit_weighted_sentiment',
                'reddit_volume', 'reddit_engagement',
                'reddit_positive_ratio']] = 0

        # ---- Merge GOOGLE TRENDS ----
        if trends_df is not None and not trends_df.empty:
            trends_df['Date'] = pd.to_datetime(trends_df['Date']).dt.tz_localize(None)
            df = df.merge(trends_df[['Date', 'search_interest']], on='Date', how='left')
            df['search_interest'] = df['search_interest'].ffill().fillna(0)
        else:
            df['search_interest'] = 0

        # ---- Fill missing sentiment values safely ----
        sentiment_cols = [
            col for col in df.columns
            if any(x in col for x in ['sentiment', 'reddit', 'news'])
        ]
        df[sentiment_cols] = df[sentiment_cols].fillna(0)

        # ---- Create lagged sentiment features (CRITICAL) ----
        df = self.create_lagged_features(df, sentiment_cols, lags=[1, 2, 3])

        # ---- Create rolling price features ----
        df = self.create_rolling_features(df, 'close', windows=[3, 7, 14])

        # ---- Target: NEXT DAY closing price ----
        df['target'] = df['close'].shift(-1)

        # ---- Drop rows with NaNs from lagging/rolling ----
        df = df.dropna(subset=['target'])

        df = df.fillna(0)


        # ---- Store final feature columns ----
        self.feature_columns = [
            col for col in df.columns
            if col not in ['Date', 'target']
        ]

        return df

    
    def prepare_train_test_split(self, df, test_size=0.2):
        """
        Time-series split: Use last test_size portion as test set
        """
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Separate features and target
        X_train = train_df[self.feature_columns]
        y_train = train_df['target']
        X_test = test_df[self.feature_columns]
        y_test = test_df['target']
        
        return X_train, X_test, y_train, y_test, test_df