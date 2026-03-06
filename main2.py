# main.py
"""
Main pipeline for Project StockSenseAI
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
from data_scrapers.news_scraper2 import NewsScraper
from data_scrapers.trends_scraper2 import TrendsScraper
from data_scrapers.reddit_scraper2 import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
from backtest import Backtester
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
    
    # 2. News Sentiment (DAILY, DATE-ALIGNED)
    news_scraper = NewsScraper()
    news_df = news_scraper.get_daily_sentiment(stock_name)

    # 3. Google Trends
    trends_scraper = TrendsScraper()
    trends_df = trends_scraper.get_search_trends(stock_name, start_date, end_date)

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

    # 4. Reddit Sentiment (DAILY, DATE-ALIGNED)
    reddit_scraper = RedditScraper()
    reddit_df = reddit_scraper.get_daily_reddit_sentiment(stock_name)

    return stock_df, news_df, trends_df, reddit_df


def engineer_features(stock_df, news_df, trends_df, reddit_df):
    """
    Create all features
    """
    print("\n" + "="*60)
    print("⚙️  FEATURE ENGINEERING PHASE")
    print("="*60)
    
    engineer = FeatureEngineer()
    combined_df = engineer.combine_all_features(
        stock_df, news_df, trends_df, reddit_df
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
    predictions here are RETURNS (from pct_change target)
    """
    print("\n" + "="*60)
    print("📝 GENERATING PREDICTION LOG")
    print("="*60)

    # Convert predicted return back to actual price
    predicted_prices = test_df['close'].values * (1 + predictions)
    actual_prices = test_df['close'].shift(-1).values  # actual next day close

    log_df = pd.DataFrame({
        'Date': test_df['Date'].values,
        'Actual_Closing_Price': actual_prices,
        'Predicted_Closing_Price': predicted_prices
    })

    log_df.to_csv(config.PREDICTION_LOG_PATH, index=False)
    print(f"✓ Prediction log saved to {config.PREDICTION_LOG_PATH}")
    
    return log_df


def run_backtest(combined_df, predictor, engineer):
    """
    Run backtesting simulation and print results
    """
    print("\n" + "="*60)
    print("📊 BACKTESTING PHASE")
    print("="*60)

    backtester = Backtester(
        initial_capital=10000,   # Start with $10,000
        transaction_cost=0.001,  # 0.1% per trade (realistic broker cost)
        threshold=0.005          # Only trade if model predicts > +0.5% or < -0.5%
    )

    metrics, portfolio_df, trade_log = backtester.run(
        combined_df=combined_df,
        predictor=predictor,
        feature_columns=engineer.feature_columns,
        test_size=config.TEST_SIZE
    )

    # ---- Print results ----
    print("\n📈 BACKTEST RESULTS")
    print("-" * 40)
    print(f"  Confidence Threshold   : ±{metrics['Threshold_Used']*100:.2f}% (HOLD if inside band)")
    print(f"  Initial Capital        : ${metrics['Initial_Capital']:,.2f}")
    print(f"  Final Capital          : ${metrics['Final_Capital']:,.2f}")
    print()
    print(f"  Strategy Return        : {metrics['Strategy_Return_%']:+.2f}%")
    print(f"  Buy & Hold Return      : {metrics['BuyHold_Return_%']:+.2f}%")
    print(f"  Alpha (vs Buy & Hold)  : {metrics['Alpha_%']:+.2f}%")
    print()
    print(f"  Sharpe Ratio           : {metrics['Sharpe_Ratio']:.3f}")
    print(f"  Max Drawdown           : -{metrics['Max_Drawdown_%']:.2f}%")
    print()
    print(f"  Total Trades           : {metrics['Total_Trades']}")
    print(f"  Win Rate               : {metrics['Win_Rate_%']:.1f}%")
    print(f"  Avg Win per Trade      : +{metrics['Avg_Win_%']:.2f}%")
    print(f"  Avg Loss per Trade     : {metrics['Avg_Loss_%']:.2f}%")
    print(f"  Profit Factor          : {metrics['Profit_Factor']:.3f}")
    print("-" * 40)

    # ---- Interpretation ----
    print("\n💡 INTERPRETATION")
    if metrics['Alpha_%'] > 0:
        print(f"  ✅ Strategy BEAT buy & hold by {metrics['Alpha_%']:.2f}%")
    else:
        print(f"  ❌ Strategy UNDERPERFORMED buy & hold by {abs(metrics['Alpha_%']):.2f}%")

    if metrics['Sharpe_Ratio'] > 1:
        print(f"  ✅ Good risk-adjusted returns (Sharpe > 1)")
    elif metrics['Sharpe_Ratio'] > 0:
        print(f"  ⚠️  Positive but weak risk-adjusted returns (Sharpe < 1)")
    else:
        print(f"  ❌ Poor risk-adjusted returns (Sharpe < 0)")

    if metrics['Max_Drawdown_%'] < 10:
        print(f"  ✅ Low drawdown — strategy is relatively stable")
    elif metrics['Max_Drawdown_%'] < 20:
        print(f"  ⚠️  Moderate drawdown — some rough patches")
    else:
        print(f"  ❌ High drawdown ({metrics['Max_Drawdown_%']:.1f}%) — strategy had significant losing periods")

    if metrics['Profit_Factor'] > 1.5:
        print(f"  ✅ Strong profit factor — wins outweigh losses")
    elif metrics['Profit_Factor'] > 1:
        print(f"  ⚠️  Marginally profitable — slim edge")
    else:
        print(f"  ❌ Profit factor < 1 — losing more than winning")

    # ---- Save backtest log ----
    backtest_log_path = os.path.join(config.OUTPUT_DIR, "backtest_log.csv")
    portfolio_df.to_csv(backtest_log_path, index=False)
    print(f"\n✓ Backtest portfolio log saved to {backtest_log_path}")

    return metrics, portfolio_df


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
    print("PROJECT StockSense AI - UNIVERSAL SENTIMENT ENGINE")
    print("="*30 + "\n")
    
    print(f"Stock: {config.STOCK_NAME}")
    print(f"Timeline: {config.START_DATE} to {config.END_DATE}")
    print()
    
    try:
        # Create output directory
        create_output_dir()
        
        # Step 1: Fetch all data
        stock_df, news_df, trends_df, reddit_df = fetch_all_data(
            config.STOCK_NAME,
            config.START_DATE,
            config.END_DATE
        )
        
        # Step 2: Engineer features
        combined_df, engineer = engineer_features(
            stock_df, news_df, trends_df, reddit_df
        )

        # Save processed features
        save_processed_features(combined_df)
        
        # Step 3: Train and evaluate
        predictor, metrics, predictions, test_df = train_and_evaluate(
            combined_df, engineer
        )
        
        # Step 4: Generate prediction log
        log_df = generate_prediction_log(test_df, predictions)

        # Step 5: Run backtest
        backtest_metrics, portfolio_df = run_backtest(combined_df, predictor, engineer)
        
        # Final summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

        predicted_price = test_df['close'].values[-1] * (1 + predictions[-1])
        actual_price = test_df['close'].values[-1] * (1 + test_df['target'].values[-1])
        print(f"\nNext day closing price prediction : ${predicted_price:.2f}")
        print(f"Actual closing price              : ${actual_price:.2f}")
        print(f"Prediction error                  : ${abs(predicted_price - actual_price):.2f}")
        
        print("\n📁 Output files:")
        print(f"  - {config.FEATURE_CSV_PATH}")
        print(f"  - {config.PREDICTION_LOG_PATH}")
        print(f"  - {config.MODEL_PATH}")
        print(f"  - {os.path.join(config.OUTPUT_DIR, 'backtest_log.csv')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)