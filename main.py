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

from data_scrapers.stock_scraper   import StockScraper
from data_scrapers.news_scraper    import NewsScraper
from data_scrapers.trends_scraper  import TrendsScraper
from data_scrapers.reddit_scraper  import RedditScraper
from features.feature_engineering  import FeatureEngineer
from model.predictor               import StockPredictor
from Strategy.backtest             import Backtester
from Strategy.stop_loss            import make_stop_loss_config
import config


def create_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def fetch_all_data(stock_name, start_date, end_date):
    print("=" * 60)
    print("📥 DATA COLLECTION PHASE")
    print("=" * 60)

    # 1. Stock data
    stock_scraper = StockScraper(stock_name)
    stock_df      = stock_scraper.fetch_historical_data(start_date, end_date)
    if stock_df.empty:
        raise ValueError("Failed to fetch stock data!")
    stock_df = stock_scraper.calculate_technical_indicators(stock_df)

    # 2. News sentiment
    news_scraper = NewsScraper()
    news_df      = news_scraper.get_daily_sentiment(stock_name)

    # 3. Google Trends — TrendsScraper handles .NS/.BO stripping and
    #    yfinance company-name fallback internally, no alias dict needed
    trends_scraper = TrendsScraper()
    trends_df      = trends_scraper.get_search_trends(
        stock_name, start_date, end_date)

    # 4. Reddit sentiment — RedditScraper adds Indian subreddits
    #    automatically for .NS/.BO tickers
    reddit_scraper = RedditScraper()
    reddit_df      = reddit_scraper.get_daily_reddit_sentiment(stock_name)

    return stock_df, news_df, trends_df, reddit_df


def engineer_features(stock_df, news_df, trends_df, reddit_df, stock_name):
    print("\n" + "=" * 60)
    print("⚙️  FEATURE ENGINEERING PHASE")
    print("=" * 60)

    engineer    = FeatureEngineer()
    combined_df = engineer.combine_all_features(
        stock_df, news_df, trends_df, reddit_df,
        ticker=stock_name,           # enables Indian circuit-breaker features
    )

    print(f"✓ Created {len(engineer.feature_columns)} features")
    print(f"✓ Dataset shape: {combined_df.shape}")
    return combined_df, engineer


def train_and_evaluate(combined_df, engineer):
    print("\n" + "=" * 60)
    print("🎯 MODEL TRAINING PHASE")
    print("=" * 60)

    X_train, X_test, y_train, y_test, test_df = \
        engineer.prepare_train_test_split(combined_df, test_size=config.TEST_SIZE)

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    predictor = StockPredictor()
    predictor.train(X_train, y_train)
    metrics, predictions = predictor.evaluate(X_test, y_test)
    predictor.save_model(config.MODEL_PATH)

    return predictor, metrics, predictions, test_df


def generate_prediction_log(test_df, predictions):
    print("\n" + "=" * 60)
    print("📝 GENERATING PREDICTION LOG")
    print("=" * 60)

    predicted_prices = test_df['close'].values * (1 + predictions)
    actual_prices    = test_df['close'].shift(-1).values

    log_df = pd.DataFrame({
        'Date':                    test_df['Date'].values,
        'Actual_Closing_Price':    actual_prices,
        'Predicted_Closing_Price': predicted_prices,
    })
    log_df.to_csv(config.PREDICTION_LOG_PATH, index=False)
    print(f"✓ Prediction log saved to {config.PREDICTION_LOG_PATH}")
    return log_df


def run_backtest(combined_df, predictor, engineer):
    print("\n" + "=" * 60)
    print("📊 BACKTESTING PHASE")
    print("=" * 60)

    sl_config = make_stop_loss_config(
        mode="trailing",
        trailing_pct=0.03,
        hard_floor_pct=0.05,
    )

    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        threshold=0.005,
        sl_config=sl_config,
    )

    metrics, portfolio_df, trade_log = backtester.run(
        combined_df=combined_df,
        predictor=predictor,
        feature_columns=engineer.feature_columns,
        test_size=config.TEST_SIZE,
    )

    print("\n📈 BACKTEST RESULTS")
    print("-" * 40)
    print(f"  Confidence Threshold   : ±{metrics['Threshold_Used']*100:.2f}%")
    print(f"  Stop Loss Mode         : {metrics['Stop_Mode']}")
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
    print(f"    → ML Exits           : {metrics['ML_Exits']}")
    print(f"    → Stop Loss Exits    : {metrics['Stop_Exits']}")
    print(f"    → Final Day Exits    : {metrics['Final_Exits']}")
    print(f"  Win Rate               : {metrics['Win_Rate_%']:.1f}%")
    print(f"  Avg Win per Trade      : +{metrics['Avg_Win_%']:.2f}%")
    print(f"  Avg Loss per Trade     : {metrics['Avg_Loss_%']:.2f}%")
    print(f"  Profit Factor          : {metrics['Profit_Factor']:.3f}")
    print("-" * 40)

    print("\n💡 INTERPRETATION")
    if metrics['Alpha_%'] > 0:
        print(f"  ✅ Strategy BEAT buy & hold by {metrics['Alpha_%']:.2f}%")
    else:
        print(f"  ❌ Strategy UNDERPERFORMED buy & hold by "
              f"{abs(metrics['Alpha_%']):.2f}%")
    if metrics['Sharpe_Ratio'] > 1:
        print("  ✅ Good risk-adjusted returns (Sharpe > 1)")
    elif metrics['Sharpe_Ratio'] > 0:
        print("  ⚠️  Positive but weak risk-adjusted returns (Sharpe < 1)")
    else:
        print("  ❌ Poor risk-adjusted returns (Sharpe < 0)")
    if metrics['Max_Drawdown_%'] < 10:
        print("  ✅ Low drawdown — strategy is relatively stable")
    elif metrics['Max_Drawdown_%'] < 20:
        print("  ⚠️  Moderate drawdown — some rough patches")
    else:
        print(f"  ❌ High drawdown ({metrics['Max_Drawdown_%']:.1f}%)")
    if metrics['Profit_Factor'] > 1.5:
        print("  ✅ Strong profit factor — wins outweigh losses")
    elif metrics['Profit_Factor'] > 1:
        print("  ⚠️  Marginally profitable — slim edge")
    else:
        print("  ❌ Profit factor < 1 — losing more than winning")

    backtest_log_path = os.path.join(config.OUTPUT_DIR, "backtest_log.csv")
    portfolio_df.to_csv(backtest_log_path, index=False)
    print(f"\n✓ Backtest log saved to {backtest_log_path}")

    return metrics, portfolio_df


def save_processed_features(combined_df):
    combined_df.to_csv(config.FEATURE_CSV_PATH, index=False)
    print(f"✓ Processed features saved to {config.FEATURE_CSV_PATH}")


def main():
    print("\n" + "🚀" * 30)
    print("PROJECT StockSense AI - UNIVERSAL SENTIMENT ENGINE")
    print("=" * 30 + "\n")
    print(f"Stock:    {config.STOCK_NAME}")
    print(f"Timeline: {config.START_DATE} to {config.END_DATE}\n")

    try:
        create_output_dir()

        stock_df, news_df, trends_df, reddit_df = fetch_all_data(
            config.STOCK_NAME, config.START_DATE, config.END_DATE)

        combined_df, engineer = engineer_features(
            stock_df, news_df, trends_df, reddit_df, config.STOCK_NAME)

        save_processed_features(combined_df)

        predictor, metrics, predictions, test_df = train_and_evaluate(
            combined_df, engineer)

        generate_prediction_log(test_df, predictions)

        run_backtest(combined_df, predictor, engineer)

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        predicted_price = test_df['close'].values[-1] * (1 + predictions[-1])
        actual_price    = (test_df['close'].values[-1] *
                           (1 + test_df['target'].values[-1]))
        print(f"\nNext day prediction : ${predicted_price:.2f}")
        print(f"Actual close        : ${actual_price:.2f}")
        print(f"Prediction error    : ${abs(predicted_price - actual_price):.2f}")

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