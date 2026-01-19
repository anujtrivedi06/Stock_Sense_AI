"""
Live Dashboard for Mystery Stock Prediction
Phase 3: Mystery Stock (Offline)
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

from data_scrapers.stock_scraper import StockScraper
from data_scrapers.news_scraper2 import NewsScraper
from data_scrapers.trends_scraper2 import TrendsScraper
from data_scrapers.reddit_scraper2 import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
import config

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Project StockSenseAI – Mystery Stock",
    page_icon="📈",
    layout="wide"
)

# ---------------- HEADER ---------------- #
st.title("🔮 Project StockSenseAI – Universal Sentiment Engine")
st.caption("Phase 3: Mystery Stock – Live Prediction Dashboard")
st.markdown("---")

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Mystery Stock Ticker", value=config.STOCK_NAME)

    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()

    st.markdown("---")
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    st.info(f"Last Updated: {last_updated}")

# ---------------- DATA FETCHING ---------------- #
@st.cache_data(ttl=1800)
def load_stock_data(ticker, days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    scraper = StockScraper(ticker)
    df = scraper.fetch_historical_data(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    df = scraper.calculate_technical_indicators(df)
    return df


def fetch_live_sentiment(ticker):
    news_scraper = NewsScraper()
    news_df = news_scraper.get_daily_sentiment(ticker)

    reddit_scraper = RedditScraper()
    reddit_df = reddit_scraper.get_daily_reddit_sentiment(ticker)

    return news_df, reddit_df


def fetch_trends(ticker):
    trends_scraper = TrendsScraper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    return trends_scraper.get_search_trends(
        ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )


def make_prediction(ticker):
    predictor = StockPredictor()
    predictor.load_model(config.MODEL_PATH)

    stock_df = load_stock_data(ticker)
    news_df, reddit_df = fetch_live_sentiment(ticker)
    trends_df = fetch_trends(ticker)

    engineer = FeatureEngineer()
    combined_df = engineer.combine_all_features(
        stock_df, news_df, trends_df, reddit_df
    )

    latest_features = combined_df[engineer.feature_columns].iloc[-1:]
    prediction = predictor.predict(latest_features)[0]

    current_price = stock_df["close"].iloc[-1]

    return prediction, current_price, news_df, reddit_df, stock_df


# ---------------- MAIN EXECUTION ---------------- #
with st.spinner("🔍 Fetching live market & sentiment data..."):
    try:
        prediction, current_price, news_df, reddit_df, stock_df = make_prediction(ticker)
    except Exception as e:
        st.error("❌ Error generating prediction. Ensure model is trained.")
        st.stop()

# ---------------- METRICS ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mystery Stock", ticker)

with col2:
    st.metric("Current Price", f"${current_price:.2f}")

with col3:
    st.metric("Predicted Next-Day Close", f"${prediction:.2f}")

price_change = prediction - current_price
price_change_pct = (price_change / current_price) * 100

st.markdown("---")

col4, col5, col6 = st.columns(3)
col4.metric("Expected Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
col5.metric("Signal", "🟢 BUY" if price_change > 0 else "🔴 SELL")
col6.metric("Confidence", f"{min(100, abs(price_change_pct)*10):.1f}%")

# ---------------- SENTIMENT ---------------- #
st.markdown("---")
st.header("📰 Live Sentiment Snapshot")

col7, col8 = st.columns(2)

with col7:
    st.subheader("News Sentiment")
    if not news_df.empty:
        st.metric("Avg Sentiment", f"{news_df['avg_sentiment'].iloc[-1]:.3f}")
        st.metric("News Volume", int(news_df['news_volume'].iloc[-1]))
    else:
        st.info("No recent news sentiment")

with col8:
    st.subheader("Reddit Sentiment")
    if not reddit_df.empty:
        st.metric("Avg Sentiment", f"{reddit_df['reddit_avg_sentiment'].iloc[-1]:.3f}")
        st.metric("Post Volume", int(reddit_df['reddit_volume'].iloc[-1]))
    else:
        st.info("No recent Reddit sentiment")

# ---------------- PRICE CHART ---------------- #
st.markdown("---")
st.header("📈 Recent Price Movement")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["close"],
    mode="lines",
    name="Closing Price"
))

next_day = stock_df["Date"].iloc[-1] + timedelta(days=1)
fig.add_trace(go.Scatter(
    x=[stock_df["Date"].iloc[-1], next_day],
    y=[current_price, prediction],
    mode="lines+markers",
    name="Prediction",
    line=dict(dash="dash")
))

fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption(f"Prediction generated live • {last_updated}")
st.caption("Project StockSenseAI – Phase 3 Submission")
