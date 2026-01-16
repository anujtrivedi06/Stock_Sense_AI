# dashboard/app.py
"""
Live Dashboard for Mystery Stock Prediction
Deploy this to Streamlit Cloud for the competition
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from data_scrapers.stock_scraper import StockScraper
from data_scrapers.news_scraper import NewsScraper
from data_scrapers.trends_scraper import TrendsScraper
from data_scrapers.reddit_scraper import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
import config

# Page configuration
st.set_page_config(
    page_title="Project Kassandra - Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data_cached(stock_name, days=365):
    """Cache data loading"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    stock_scraper = StockScraper(stock_name)
    stock_df = stock_scraper.fetch_historical_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    stock_df = stock_scraper.calculate_technical_indicators(stock_df)
    
    return stock_df

def fetch_live_sentiment(stock_name):
    """Fetch live sentiment data"""
    news_scraper = NewsScraper()
    sentiment = news_scraper.get_aggregated_sentiment(stock_name)
    
    reddit_scraper = RedditScraper()
    reddit_sentiment = reddit_scraper.get_reddit_sentiment(stock_name)
    
    return sentiment, reddit_sentiment

def make_prediction(stock_name):
    """Make prediction for next day"""
    # Load model
    predictor = StockPredictor()
    
    try:
        predictor.load_model(config.MODEL_PATH)
    except:
        st.error("Model not found! Please train the model first using main.py")
        return None, None, None
    
    # Get recent data
    stock_df = load_data_cached(stock_name, days=365)
    sentiment, reddit_sentiment = fetch_live_sentiment(stock_name)
    
    # Get trends
    trends_scraper = TrendsScraper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    trends_df = trends_scraper.get_search_trends(
        stock_name,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Engineer features
    engineer = FeatureEngineer()
    combined_df = engineer.combine_all_features(
        stock_df, sentiment, trends_df, reddit_sentiment
    )
    
    # Get latest features
    latest_features = combined_df[engineer.feature_columns].iloc[-1:].fillna(0)
    
    # Make prediction
    prediction = predictor.predict(latest_features)[0]
    current_price = stock_df['close'].iloc[-1]
    
    return prediction, current_price, sentiment, reddit_sentiment

def main():
    # Header
    st.title("🔮 Project Kassandra: Universal Sentiment Engine")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        stock_name = st.text_input("Stock Ticker", value=config.STOCK_NAME)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
        
        st.markdown("---")
        st.markdown("### 📊 Last Updated")
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(last_updated)
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Mystery Stock")
        st.markdown(f'<p class="big-font">{stock_name}</p>', unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("🔍 Analyzing market and sentiment data..."):
        prediction, current_price, sentiment, reddit_sentiment = make_prediction(stock_name)
    
    if prediction is not None:
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        with col2:
            st.markdown("### 💰 Current Price")
            st.markdown(f'<p class="big-font">${current_price:.2f}</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("### 🔮 Predicted Price")
            st.markdown(f'<p class="big-font">${prediction:.2f}</p>', unsafe_allow_html=True)
        
        # Prediction details
        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric(
                "Price Change",
                f"${price_change:.2f}",
                f"{price_change_pct:.2f}%"
            )
        
        with col5:
            decision = "🟢 BUY" if price_change > 0 else "🔴 SELL"
            st.metric("Decision", decision)
        
        with col6:
            confidence = min(100, abs(price_change_pct) * 10)
            st.metric("Confidence", f"{confidence:.1f}%")
        
        # Sentiment Analysis
        st.markdown("---")
        st.header("📰 Sentiment Analysis")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("News Sentiment")
            if sentiment:
                st.metric("Average Sentiment", f"{sentiment.get('avg_sentiment', 0):.3f}")
                st.metric("News Volume", sentiment.get('news_volume', 0))
                st.metric("Positive Ratio", f"{sentiment.get('positive_ratio', 0)*100:.1f}%")
        
        with col8:
            st.subheader("Reddit Sentiment")
            if reddit_sentiment:
                st.metric("Reddit Sentiment", f"{reddit_sentiment.get('reddit_avg_sentiment', 0):.3f}")
                st.metric("Post Volume", reddit_sentiment.get('reddit_volume', 0))
                st.metric("Engagement", reddit_sentiment.get('reddit_engagement', 0))
        
        # Price chart
        st.markdown("---")
        st.header("📈 Price History")
        
        stock_df = load_data_cached(stock_name, days=90)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_df['Date'],
            y=stock_df['close'],
            mode='lines',
            name='Closing Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add prediction point
        next_date = stock_df['Date'].iloc[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[stock_df['Date'].iloc[-1], next_date],
            y=[current_price, prediction],
            mode='lines+markers',
            name='Prediction',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"{stock_name} Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("*Powered by Project Kassandra - Universal Sentiment Engine*")
        st.markdown(f"*Last Updated: {last_updated}*")

if __name__ == "__main__":
    main()