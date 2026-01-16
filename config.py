# config.py
"""
Configuration file for Project Kassandra
Modify STOCK_NAME and TIMELINE as needed
"""

# Stock Configuration
STOCK_NAME = "DIS"  # Change this for mystery stock
START_DATE = "2021-01-16"
END_DATE = "2026-01-16"

# API Keys (Get free keys from respective platforms)
NEWS_API_KEY = "your_newsapi_key_here"  # Get from newsapi.org
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_reddit_secret"
REDDIT_USER_AGENT = "KassandraBot/1.0"

# Feature Engineering Parameters
SENTIMENT_WINDOW = 3  # Days to aggregate sentiment
TECHNICAL_INDICATORS = ['SMA_5', 'SMA_20', 'RSI', 'MACD']

# Model Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
PREDICTION_DAYS = 1  # Predict next day

# Paths
OUTPUT_DIR = "outputs/"
FEATURE_CSV_PATH = "outputs/processed_features.csv"
PREDICTION_LOG_PATH = "outputs/prediction_log.csv"
MODEL_PATH = "outputs/trained_model.pkl"