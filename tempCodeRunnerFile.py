# main.py
"""
Main pipeline for Project Kassandra
Complete end-to-end stock prediction system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_scrapers.stock_scraper import StockScraper
from data_scrapers.news_scraper import NewsScraper
from data_scrapers.trends_scraper import TrendsScraper
from data_scrapers.reddit_scraper import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
import config

def create_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

def fetch_all_data(stock_name, start_date, end_date):
    """
    Fetch all data sources
    """
    print("="*60)
    print("📥 DATA COLLECTION PHASE")
    print("="*60)
    
    # 1. Stock Data
    stock_scraper = StockScraper(stock_name)
    stock_df = stock_scraper.fetch_historical_data(start_date, end_date)
    
    if stock_df.empty:
        raise ValueError("Failed to fetch stock data!")
    
    # Add technical indicators
    stock_df = stock_scraper.calculate_technical_indicators(stock_df)
    
    # 2. News Sentiment (gets current sentiment, will be lagged in feature engineering)
    news_scraper = NewsScraper()
    sentiment_dict = news_scraper.get_aggregated_sentiment(stock_name)
    
    # 3. Google Trends
    trends_scraper = TrendsScraper()
    
    # Try stock ticker first, then company name
    trends_df = trends_scraper.get_search_trends(stock_name, start_date, end_date)
    
    # If no data, try common company names
    if trends_df.empty:
        company_map = {
            'TSLA': 'Tesla',
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon'
        }
        if stock_name in company_map:
            trends_df = trends_scraper.get_search_trends(
                company_map[stock_name], start_date, end_date
            )
    
    # 4. Reddit Sentiment
    reddit_scraper = RedditScraper()
    reddit_dict = reddit_scraper.get_reddit_sentiment(stock_name)
    
    return stock_df, sentiment_dict, trends_df, reddit_dict

def engineer_features(stock_df, sentiment_dict, trends_df, reddit_dict):
    """
    Create all features
    """
    print("\n" + "="*60)
    print("⚙️  FEATURE ENGINEERING PHASE")
    print("="*60)
    
    engineer = FeatureEngineer()
    combined_df = engineer.combine_all_features(
        stock_df, sentiment_dict, trends_df, reddit_dict
    )
    
    print(f"✓ Created {len(engineer.feature_columns)} features")
    print(f"✓ Dataset shape: {combined_df.shape}")
    
    return combined_df, engineer

def train_and_evaluate(combined_df, engineer):
    """
    Train model and evaluate
    """
    print("\n" + "="*60)
    print("🎯 MODEL TRAINING PHASE")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test, test_df = engineer.prepare_train_test_split(
        combined_df, test_size=config.TEST_SIZE
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    predictor = StockPredictor()
    predictor.train(X_train, y_train)
    
    # Evaluate
    metrics, predictions = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model(config.MODEL_PATH)
    
    return predictor, metrics, predictions, test_df

def generate_prediction_log(test_df, predictions):
    """
    Generate prediction log CSV
    """
    print("\n" + "="*60)
    print("📝 GENERATING PREDICTION LOG")
    print("="*60)
    
    log_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'Actual_Closing_Price': test_df['target'].values,
        'Predicted_Closing_Price': predictions
    })
    
    log_df.to_csv(config.PREDICTION_LOG_PATH, index=False)
    print(f"✓ Prediction log saved to {config.PREDICTION_LOG_PATH}")
    
    return log_df

def save_processed_features(combined_df):
    """
    Save processed features CSV
    """
    combined_df.to_csv(config.FEATURE_CSV_PATH, index=False)
    print(f"✓ Processed features saved to {config.FEATURE_CSV_PATH}")

def main():
    """
    Main execution pipeline
    """
    print("\n" + "🚀"*30)
    print("PROJECT KASSANDRA - UNIVERSAL SENTIMENT ENGINE")
    print("🚀"*30 + "\n")
    
    print(f"Stock: {config.STOCK_NAME}")
    print(f"Timeline: {config.START_DATE} to {config.END_DATE}")
    print()
    
    try:
        # Create output directory
        create_output_dir()
        
        # Step 1: Fetch all data
        stock_df, sentiment_dict, trends_df, reddit_dict = fetch_all_data(
            config.STOCK_NAME,
            config.START_DATE,
            config.END_DATE
        )
        
        # Step 2: Engineer features
        combined_df, engineer = engineer_features(
            stock_df, sentiment_dict, trends_df, reddit_dict
        )
        
        # Save processed features
        save_processed_features(combined_df)
        
        # Step 3: Train and evaluate
        predictor, metrics, predictions, test_df = train_and_evaluate(
            combined_df, engineer
        )
        
        # Step 4: Generate prediction log
        log_df = generate_prediction_log(test_df, predictions)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nNext day closing price prediction: ${predictions[-1]:.2f}")
        print(f"Actual closing price: ${test_df['target'].values[-1]:.2f}")
        print(f"Prediction error: ${abs(predictions[-1] - test_df['target'].values[-1]):.2f}")
        
        print("\n📁 Output files:")
        print(f"  - {config.FEATURE_CSV_PATH}")
        print(f"  - {config.PREDICTION_LOG_PATH}")
        print(f"  - {config.MODEL_PATH}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)