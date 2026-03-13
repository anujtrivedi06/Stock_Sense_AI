# config.py

"""
Configuration file for Project StockSenseAI
Modify STOCK_NAME and TIMELINE as needed
"""
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file


# Stock Configuration
STOCK_NAME = "DIS"  
START_DATE = "2021-01-16"
END_DATE = "2026-02-26"

# API Keys (Get free keys from respective platforms)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Get from newsapi.org
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "StockSenseAIBot/1.0"

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