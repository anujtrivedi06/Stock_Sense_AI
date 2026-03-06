"""
Interactive Streamlit App
Train-on-demand stock prediction with per-stock caching + chart
MULTI-STOCK VERSION: Predicts 3-4 stocks simultaneously
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_scrapers.stock_scraper import StockScraper
from data_scrapers.news_scraper2 import NewsScraper
from data_scrapers.trends_scraper2 import TrendsScraper
from data_scrapers.reddit_scraper2 import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
import config


FEATURE_DIR = "outputs/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

def get_feature_path(ticker):
    return f"{FEATURE_DIR}/{ticker}.csv"

# -------------------- Paths for Cached Models --------------------
MODEL_DIR = os.path.join("outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.upper()}.pkl")


import yfinance as yf

@st.cache_data(ttl=600)
def fetch_quote_snapshot(ticker):
    """Fetch market snapshot with error handling"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        snapshot = {
            "open": info.get("open"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "year_high": info.get("fiftyTwoWeekHigh"),
            "year_low": info.get("fiftyTwoWeekLow"),
        }
        return snapshot
    except Exception as e:
        # Return empty snapshot if fetch fails
        st.warning(f"⚠️ Could not fetch real-time market data for {ticker}. Using historical data only.")
        return {
            "open": None,
            "previous_close": None,
            "day_high": None,
            "day_low": None,
            "volume": None,
            "avg_volume": None,
            "year_high": None,
            "year_low": None,
        }

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="StockSenseAI – Multi-Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 20px;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #666;
        font-weight: 500;
    }
    
    /* Cards effect for metrics */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
        padding: 15px;
    }
    
    /* Signal badges */
    .signal-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        margin: 10px 0;
    }
    
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Section dividers */
    hr {
        margin: 30px 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        color: #2c3e50;
        font-size: 20px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #212121;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Stock section headers */
    .stock-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 30px 0 20px 0;
        font-size: 24px;
        font-weight: 700;
    }
            
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSpinner"] div {
    color: #1a1a1a !important;
    font-size: 18px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Sidebar label color */
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
    color: #000000 !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)



# -------------------- Header --------------------
st.title("📈 StockSenseAI - Multi-Stock Edition")
st.markdown("<p style='font-size: 18px; color: #666; margin-top: -10px;'>AI-Powered Multi-Stock Prediction with Sentiment Analysis</p>", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    stock_input = st.text_input("📊 Stock Tickers", value="TSLA, AAPL, GOOGL", help="Enter 3-4 stock ticker symbols separated by commas (e.g., TSLA, AAPL, GOOGL, MSFT)")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("📅 Start Date", value=config.START_DATE)
    with col2:
        end_date = st.text_input("📅 End Date", value=config.END_DATE)

    st.markdown("<br>", unsafe_allow_html=True)
    
    force_retrain = st.checkbox("🔄 Force Retrain Model", value=False, help="Check to retrain model even if cache exists")
    
    train_button = st.button("🚀 Generate Predictions")

    st.markdown("---")
    st.markdown("### 🕒 Last Updated")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), language=None)
    
    st.markdown("---")
    st.markdown("""
    <div style='padding: 15px; background-color: #e8f5e9; border-radius: 8px; border-left: 4px solid #4CAF50;'>
        <b style='color: #1b5e20;'>💡 How it works:</b><br>
        <small style='color: #2e7d32;'>
        • Analyzes market data<br>
        • Processes sentiment signals<br>
        • Trains ML model per stock<br>
        • Predicts next-day prices
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 12px; background-color: #fff3e0; border-radius: 8px; border-left: 4px solid #FF9800;'>
        <b style='color: #e65100;'>⚠️ Network Issues?</b><br>
        <small style='color: #ef6c00;'>
        If you see DNS/connection errors:<br>
        • Check your internet connection<br>
        • Try disabling VPN/proxy<br>
        • Check firewall settings<br>
        • Use cached models (uncheck Force Retrain)
        </small>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Core Pipeline --------------------
def run_pipeline(stock_name, start_date, end_date, force_retrain=False):

    model_path = get_model_path(stock_name)
    feature_path = get_feature_path(stock_name)

    predictor = StockPredictor()
    engineer = FeatureEngineer()

    # -------- USE CACHE --------
    if os.path.exists(model_path) and os.path.exists(feature_path) and not force_retrain:

        predictor.load_model(model_path)

        combined_df = pd.read_csv(feature_path)
        combined_df["Date"] = pd.to_datetime(combined_df["Date"])

        engineer.feature_columns = [
            col for col in combined_df.columns if col not in ['Date', 'target', 'date']
        ]

        X_train, X_test, y_train, y_test, test_df = engineer.prepare_train_test_split(
            combined_df, test_size=config.TEST_SIZE
        )

        test_returns = predictor.predict(X_test)
        test_predictions = test_df['close'].values * (1 + test_returns)
        
        # Generate predictions for ALL data points
        all_features = combined_df[engineer.feature_columns]
        all_returns = predictor.predict(all_features)
        all_predictions = combined_df['close'].values * (1 + all_returns)

        latest_features = combined_df[engineer.feature_columns].iloc[-1:]
        latest_return = predictor.predict(latest_features)[0]
        current_price = combined_df["close"].iloc[-1]
        latest_prediction = current_price * (1 + latest_return)
        stock_df = combined_df.copy()

        return latest_prediction, current_price, stock_df, combined_df, test_df, test_predictions, all_predictions, "cached"

    # -------- OTHERWISE FULL PIPELINE --------

    stock_scraper = StockScraper(stock_name)
    stock_df = stock_scraper.fetch_historical_data(start_date, end_date)
    stock_df = stock_scraper.calculate_technical_indicators(stock_df)

    news_df = NewsScraper().get_daily_sentiment(stock_name)
    trends_df = TrendsScraper().get_search_trends(stock_name, start_date, end_date)
    reddit_df = RedditScraper().get_daily_reddit_sentiment(stock_name)

    combined_df = engineer.combine_all_features(stock_df, news_df, trends_df, reddit_df)

    combined_df.to_csv(feature_path, index=False)

    X_train, X_test, y_train, y_test, test_df = engineer.prepare_train_test_split(
        combined_df, test_size=config.TEST_SIZE
    )

    predictor.train(X_train, y_train)
    predictor.save_model(model_path)

    test_returns = predictor.predict(X_test)
    test_predictions = test_df['close'].values * (1 + test_returns)
    
    # Generate predictions for ALL data points
    all_features = combined_df[engineer.feature_columns]
    all_returns = predictor.predict(all_features)
    all_predictions = combined_df['close'].values * (1 + all_returns)

    latest_features = combined_df[engineer.feature_columns].iloc[-1:]
    latest_return = predictor.predict(latest_features)[0]
    current_price = stock_df['close'].iloc[-1]
    latest_prediction = current_price * (1 + latest_return)

    return latest_prediction, current_price, stock_df, combined_df, test_df, test_predictions, all_predictions, "trained"


def display_stock_prediction(ticker, prediction, current_price, stock_df, combined_df, test_df, test_predictions, all_predictions, snapshot):
    """Display prediction results for a single stock"""
    
    delta = prediction - current_price
    delta_pct = (delta / current_price) * 100
    
    # Stock header
    st.markdown(f"<div class='stock-header'>📊 {ticker}</div>", unsafe_allow_html=True)
    
    # -------------------- Main Prediction Cards --------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Predicted Price (Next Day)",
            value=f"${prediction:.2f}",
            delta=f"{delta:.2f} ({delta_pct:.2f}%)"
        )
    
    with col3:
        signal = "BUY 📈" if delta > 0 else "SELL 📉"
        signal_class = "signal-buy" if delta > 0 else "signal-sell"
        
        st.markdown("<p style='color: #1a1a1a; font-weight: bold;'>Trading Signal</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-badge {signal_class}'>{signal}</div>", unsafe_allow_html=True)
        st.caption("Based on ML prediction")

    # -------------------- Market Snapshot --------------------
    with st.expander(f"📊 Market Snapshot - {ticker}", expanded=False):
        # First row
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Open", f"${snapshot['open']:.2f}" if snapshot['open'] else "N/A")
        with c2:
            st.metric("Previous Close", f"${snapshot['previous_close']:.2f}" if snapshot['previous_close'] else "N/A")
        with c3:
            st.metric("Day High", f"${snapshot['day_high']:.2f}" if snapshot['day_high'] else "N/A")
        with c4:
            st.metric("Day Low", f"${snapshot['day_low']:.2f}" if snapshot['day_low'] else "N/A")

        # Second row
        c5, c6, c7, c8 = st.columns(4)

        with c5:
            st.metric("52W High", f"${snapshot['year_high']:.2f}" if snapshot['year_high'] else "N/A")
        with c6:
            st.metric("52W Low", f"${snapshot['year_low']:.2f}" if snapshot['year_low'] else "N/A")
        with c7:
            st.metric("Volume", f"{snapshot['volume']:,}" if snapshot['volume'] else "N/A")
        with c8:
            st.metric("Avg Volume", f"{snapshot['avg_volume']:,}" if snapshot['avg_volume'] else "N/A")

    # -------------------- Chart --------------------
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=combined_df['Date'],
        y=combined_df['close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ))

    # Full predicted line for entire dataset
    # Debug: Check if predictions are valid
    valid_predictions = all_predictions[~np.isnan(all_predictions)] if len(all_predictions) > 0 else []
    
    if len(valid_predictions) > 0:
        fig.add_trace(go.Scatter(
            x=combined_df['Date'],
            y=all_predictions,
            mode='lines',
            name=f'Model Predictions (std={np.std(all_predictions):.2f})',
            line=dict(color='#FF9800', width=2, dash='dash')
        ))
    else:
        st.warning(f"⚠️ No valid predictions generated for {ticker}")

    # Next day prediction
    next_day = stock_df['Date'].iloc[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[stock_df['Date'].iloc[-1], next_day],
        y=[current_price, prediction],
        mode='lines+markers',
        name='Next Day Forecast',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10, color='#4CAF50')
    ))

    fig.update_layout(
        title=f"{ticker} Price Analysis",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title="Date",
            title_font=dict(color='#1a1a1a'),
            tickfont=dict(color='#1a1a1a'),
            showgrid=True,
            gridcolor='#f0f0f0',
            linecolor='#e0e0e0'
        ),
        yaxis=dict(
            title="Price ($)",
            title_font=dict(color='#1a1a1a'),
            tickfont=dict(color='#1a1a1a'),
            showgrid=True,
            gridcolor='#f0f0f0',
            linecolor='#e0e0e0'
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#333")
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Debug information
    with st.expander(f"🔍 Debug Info - {ticker}"):
        st.write(f"**Prediction Statistics:**")
        st.write(f"- Number of predictions: {len(all_predictions)}")
        st.write(f"- Min: ${np.min(all_predictions):.2f}")
        st.write(f"- Max: ${np.max(all_predictions):.2f}")
        st.write(f"- Mean: ${np.mean(all_predictions):.2f}")
        st.write(f"- Std Dev: ${np.std(all_predictions):.2f}")
        
        if np.std(all_predictions) < 1.0:
            st.error("⚠️ Predictions have very low variance - model may not be learning!")
        
        st.write(f"\n**Current vs Predicted:**")
        st.write(f"- Current price: ${current_price:.2f}")
        st.write(f"- Next day prediction: ${prediction:.2f}")
        st.write(f"- Change: ${prediction - current_price:.2f} ({((prediction - current_price) / current_price * 100):.2f}%)")
    
    st.markdown("---")


# -------------------- UI Output --------------------
if train_button:
    # Parse stock tickers
    stock_tickers = [ticker.strip().upper() for ticker in stock_input.split(",") if ticker.strip()]
    
    # Validate input
    if len(stock_tickers) < 3 or len(stock_tickers) > 4:
        st.error("⚠️ Please enter 3-4 stock tickers separated by commas.")
    else:
        st.markdown(f"<p style='color: #1a1a1a; background-color: #e3f2fd; padding: 12px; border-radius: 8px; border-left: 4px solid #2196F3;'>🔄 Processing {len(stock_tickers)} stocks: {', '.join(stock_tickers)}</p>", unsafe_allow_html=True)
        
        # Store results for all stocks
        all_results = {}
        
        # Process each stock
        for idx, ticker in enumerate(stock_tickers):
            st.markdown(f"### Processing {ticker} ({idx+1}/{len(stock_tickers)})...")
            
            with st.spinner(f"🔄 Generating prediction for {ticker}..."):
                try:
                    result = run_pipeline(
                        ticker,
                        start_date,
                        end_date,
                        force_retrain=force_retrain
                    )
                    
                    prediction, current_price, stock_df, combined_df, test_df, test_predictions, all_predictions, status = result
                    
                    # Try to fetch snapshot, but don't fail if it doesn't work
                    try:
                        snapshot = fetch_quote_snapshot(ticker)
                    except Exception as snapshot_error:
                        st.warning(f"⚠️ Could not fetch real-time data for {ticker}. Using cached data only.")
                        snapshot = {
                            "open": None, "previous_close": None, "day_high": None, 
                            "day_low": None, "volume": None, "avg_volume": None,
                            "year_high": None, "year_low": None
                        }
                    
                    all_results[ticker] = {
                        'prediction': prediction,
                        'current_price': current_price,
                        'stock_df': stock_df,
                        'combined_df': combined_df,
                        'test_df': test_df,
                        'test_predictions': test_predictions,
                        'all_predictions': all_predictions,
                        'snapshot': snapshot,
                        'status': status
                    }
                    
                    status_emoji = "✅" if status == "cached" else "🔄"
                    status_text = "Using cached model" if status == "cached" else "Trained new model"
                    
                    # Custom styled success message with black text
                    st.markdown(
                        f"<p style='color: #1a1a1a; background-color: #e8f5e9; padding: 12px; "
                        f"border-radius: 8px; border-left: 4px solid #4CAF50;'>"
                        f"{status_emoji} <b>{ticker}</b>: {status_text}</p>",
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing {ticker}: {str(e)}")
                    # Log the error but continue with other stocks
                    import traceback
                    with st.expander(f"🔍 Error details for {ticker}"):
                        st.code(traceback.format_exc())
        
        # Display summary comparison
        if all_results:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("## 📊 Summary Comparison")
            
            cols = st.columns(len(all_results))
            for idx, (ticker, data) in enumerate(all_results.items()):
                with cols[idx]:
                    delta = data['prediction'] - data['current_price']
                    delta_pct = (delta / data['current_price']) * 100
                    
                    st.metric(
                        label=f"{ticker}",
                        value=f"${data['prediction']:.2f}",
                        delta=f"{delta_pct:.2f}%"
                    )
                    
                    signal = "BUY 📈" if delta > 0 else "SELL 📉"
                    signal_class = "signal-buy" if delta > 0 else "signal-sell"
                    st.markdown(f"<div class='signal-badge {signal_class}' style='font-size: 12px; padding: 5px 15px;'>{signal}</div>", unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Display detailed results for each stock
            st.markdown("## 📈 Detailed Predictions")
            
            for ticker, data in all_results.items():
                display_stock_prediction(
                    ticker,
                    data['prediction'],
                    data['current_price'],
                    data['stock_df'],
                    data['combined_df'],
                    data['test_df'],
                    data['test_predictions'],
                    data['all_predictions'],
                    data['snapshot']
                )
            
            # Model info
            with st.expander("ℹ️ Model Information"):
                st.markdown("""
                <div style='color: #1a1a1a;'>
                <p><b>Features Used:</b></p>
                <ul>
                <li>Technical indicators (RSI, MACD, Bollinger Bands)</li>
                <li>News sentiment analysis</li>
                <li>Google Trends data</li>
                <li>Reddit sentiment scores</li>
                </ul>
                
                <p><b>Model Type:</b> Machine Learning Ensemble (per stock)</p>
                
                <p><b>Training Data:</b> Historical price and sentiment data from specified date range</p>
                
                <p><b>Caching:</b> Each stock has its own cached model for faster subsequent predictions</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # -------------------- Welcome Screen --------------------
    st.markdown("""
    <div style='padding: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; text-align: center; margin: 50px 0;'>
        <h2 style='color: #1a1a1a; margin-bottom: 20px;'>Welcome to StockSenseAI - Multi-Stock Edition</h2>
        <p style='color: #2c3e50; font-size: 18px;'>Enter 3-4 stock tickers in the sidebar and click <b>Generate Predictions</b> to get started.</p>
        <p style='color: #34495e; margin-top: 20px;'>Our AI analyzes market data, news sentiment, and social trends to predict next-day stock prices for multiple stocks simultaneously.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #2196F3;'>
            <h3 style='color: #2196F3;'>📊 Multi-Stock Analysis</h3>
            <p style='color: #2c3e50;'>Analyze and predict 3-4 stocks simultaneously with independent models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #FF9800;'>
            <h3 style='color: #FF9800;'>🤖 Smart Caching</h3>
            <p style='color: #2c3e50;'>Per-stock model caching for faster subsequent predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #4CAF50;'>
            <h3 style='color: #4CAF50;'>📈 Comprehensive View</h3>
            <p style='color: #2c3e50;'>Compare predictions, signals, and charts across multiple stocks</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    <p><b>StockSenseAI</b> | Multi-Stock Edition | Train-on-Demand | Per-Stock Model Caching</p>
</div>
""", unsafe_allow_html=True)