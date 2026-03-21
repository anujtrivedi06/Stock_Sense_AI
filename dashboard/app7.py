"""
Interactive Streamlit App
Train-on-demand stock prediction with per-stock caching + chart
MULTI-STOCK VERSION: Predicts 3-4 stocks simultaneously
Includes: Stop Loss, Martingale Strategy Backtesting
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

from stop_loss import StopLossManager, StopLossConfig, compute_atr, make_stop_loss_config
from martingale import MartingaleBacktester
from data_scrapers.stock_scraper import StockScraper
from data_scrapers.news_scraper2 import NewsScraper
from data_scrapers.trends_scraper2 import TrendsScraper
from data_scrapers.reddit_scraper2 import RedditScraper
from features.feature_engineering import FeatureEngineer
from model.predictor import StockPredictor
from model.general_predictor import GeneralPredictor
import config

GENERAL_MODEL_PATH = getattr(config, 'GENERAL_MODEL_PATH',
                              'outputs/models/general_model.pkl')


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
        st.warning(f"⚠️ Could not fetch real-time market data for {ticker}. Using historical data only.")
        return {
            "open": None, "previous_close": None, "day_high": None,
            "day_low": None, "volume": None, "avg_volume": None,
            "year_high": None, "year_low": None,
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
    .main { background-color: #ffffff; }

    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }

    h1 {
        color: #1a1a1a; font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #4CAF50;
        margin-bottom: 20px;
    }
    h2 { color: #2c3e50; font-weight: 600; margin-top: 30px; margin-bottom: 20px; }
    h3 { color: #34495e; font-weight: 600; }

    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1a1a1a; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #666; font-weight: 500; }

    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .stButton>button {
        background-color: #4CAF50; color: white; font-weight: 600;
        border: none; border-radius: 8px; padding: 12px 24px;
        width: 100%; transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .stTextInput>div>div>input {
        border-radius: 8px; border: 1px solid #e0e0e0; padding: 10px;
    }
    .stAlert { border-radius: 10px; border-left: 4px solid; padding: 15px; }

    .signal-badge {
        display: inline-block; padding: 8px 20px; border-radius: 20px;
        font-weight: 600; font-size: 16px; margin: 10px 0;
    }
    .signal-buy  { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .signal-sell { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

    hr { margin: 30px 0; border: none; border-top: 1px solid #e0e0e0; }

    [data-testid="stSidebar"] h2 {
        color: #2c3e50; font-size: 20px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px; margin-bottom: 20px;
    }

    .footer {
        text-align: center; color: #212121;
        padding: 20px; margin-top: 50px;
        border-top: 1px solid #e0e0e0;
    }

    .stock-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 15px 20px; border-radius: 10px;
        margin: 30px 0 20px 0; font-size: 24px; font-weight: 700;
    }

    /* Martingale table styling */
    .mg-win  { color: #155724; font-weight: 600; }
    .mg-loss { color: #721c24; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSpinner"] div {
    color: #1a1a1a !important; font-size: 18px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
    color: #000000 !important; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Force alert text to black */
div[data-testid="stAlert"] p {
    color: #000000 !important;
}

/* Also cover markdown elements inside alerts */
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #000000 !important;
}

/* Fix expander titles like Market Snapshot */
[data-testid="stExpander"] summary {
    color: #000000 !important;
    font-weight: 600;
}

/* Fix headings like Martingale Strategy Backtest */
h4, h3 {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.title("📈 StockSenseAI - Multi-Stock Edition")
st.markdown(
    "<p style='font-size: 18px; color: #666; margin-top: -10px;'>"
    "AI-Powered Multi-Stock Prediction with Sentiment Analysis</p>",
    unsafe_allow_html=True
)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    stock_input = st.text_input(
        "📊 Stock Tickers", value="TSLA, AAPL, GOOGL",
        help="Enter 3-4 stock ticker symbols separated by commas"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("📅 Start Date", value=config.START_DATE)
    with col2:
        end_date = st.text_input("📅 End Date", value=config.END_DATE)

    st.markdown("<br>", unsafe_allow_html=True)
    force_retrain = st.checkbox(
        "🔄 Force Retrain Model", value=False,
        help="Check to retrain model even if cache exists"
    )

    # ── Stop Loss Settings ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛡️ Stop Loss Settings")

    sl_mode = st.selectbox(
        "Stop Loss Mode",
        options=["fixed", "trailing", "atr"],
        format_func=lambda x: {
            "fixed": "Fixed %", "trailing": "Trailing %", "atr": "ATR-Based"
        }[x],
        help="Fixed: static %; Trailing: follows price up; ATR: volatility-adjusted"
    )

    if sl_mode == "fixed":
        sl_fixed_pct    = st.slider("Stop Loss %", 0.5, 10.0, 2.0, 0.5) / 100
        sl_trailing_pct = 0.02
        sl_atr_mult     = 2.0
    elif sl_mode == "trailing":
        sl_trailing_pct = st.slider("Trailing Stop %", 0.5, 10.0, 2.0, 0.5) / 100
        sl_fixed_pct    = 0.02
        sl_atr_mult     = 2.0
    else:
        sl_atr_mult     = st.slider("ATR Multiplier", 0.5, 5.0, 2.0, 0.5)
        sl_fixed_pct    = 0.02
        sl_trailing_pct = 0.02

    sl_hard_floor = st.slider("Hard Floor (max loss %)", 0, 20, 5, 1) / 100

    sl_config = make_stop_loss_config(
        mode=sl_mode,
        fixed_pct=sl_fixed_pct,
        trailing_pct=sl_trailing_pct,
        atr_multiplier=sl_atr_mult,
        hard_floor_pct=sl_hard_floor
    )

    # ── Martingale Settings ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎲 Martingale Settings")

    mg_enabled = st.checkbox(
        "Enable Martingale Backtest", value=True,
        help="Run Martingale strategy simulation after predictions are generated"
    )
    mg_capital    = st.number_input("Starting Capital ($)", min_value=1000, max_value=1_000_000,
                                     value=10_000, step=1000)
    mg_base_bet   = st.slider("Base Bet % of Capital", 1, 20, 5, 1,
                               help="Percentage of capital used as the base bet size") / 100
    mg_max_mult   = st.selectbox("Max Doubling Cap", [4, 8, 16, 32], index=1,
                                  help="Maximum multiplier before bet size is capped")
    mg_threshold  = st.slider("Signal Threshold %", 0, 200, 50, 10,
                               help="Minimum predicted move (in basis points) to enter a trade") / 10000
    mg_cost       = st.slider("Transaction Cost %", 0, 50, 10, 5,
                               help="Brokerage cost per trade in basis points") / 10000

    train_button = st.button("🚀 Generate Predictions")

    st.markdown("---")
    st.markdown("### 🕒 Last Updated")
    st.code(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), language=None)

    st.markdown("---")
    st.markdown("""
    <div style='padding: 15px; background-color: #e8f5e9; border-radius: 8px;
                border-left: 4px solid #4CAF50;'>
        <b style='color: #1b5e20;'>💡 How it works:</b><br>
        <small style='color: #2e7d32;'>
        • Analyzes market data<br>
        • Processes sentiment signals<br>
        • Trains ML model per stock<br>
        • Predicts next-day prices<br>
        • Runs Stop-Loss & Martingale analysis
        </small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 12px; background-color: #fff3e0; border-radius: 8px;
                border-left: 4px solid #FF9800;'>
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
@st.cache_resource
def load_general_model():
    """Load the general model once and cache for the entire session."""
    gp = GeneralPredictor()
    if gp.is_available(GENERAL_MODEL_PATH):
        gp.load_model(GENERAL_MODEL_PATH)
        return gp
    return None


def _fetch_sentiment(stock_name, start_date, end_date):
    """Fetch news, trends, reddit in parallel. Returns dict of DataFrames."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    company_map = getattr(config, 'COMPANY_ALIASES', {})

    def fetch_news():
        return NewsScraper().get_daily_sentiment(stock_name)
    def fetch_trends():
        s  = TrendsScraper()
        df = s.get_search_trends(stock_name, start_date, end_date)
        if df.empty and stock_name in company_map:
            df = s.get_search_trends(company_map[stock_name], start_date, end_date)
        return df
    def fetch_reddit():
        return RedditScraper().get_daily_reddit_sentiment(stock_name)

    result = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fn): name for name, fn in [
            ('news', fetch_news), ('trends', fetch_trends), ('reddit', fetch_reddit)
        ]}
        for f in as_completed(futs):
            name = futs[f]
            try:    result[name] = f.result()
            except: result[name] = pd.DataFrame()
    return result


def run_pipeline(stock_name, start_date, end_date, force_retrain=False):
    """
    Prediction pipeline — uses general model by default.

    Priority order:
      1. General model  → instant, no training needed
      2. Per-stock cache → fast if cache exists
      3. Per-stock train → slowest fallback
    """
    general_model = load_general_model()

    if general_model is not None and not force_retrain:
        # ── General model path ────────────────────────────────────────
        stock_scraper = StockScraper(stock_name)
        stock_df      = stock_scraper.fetch_historical_data(start_date, end_date)
        stock_df      = stock_scraper.calculate_technical_indicators(stock_df)

        sentiment   = _fetch_sentiment(stock_name, start_date, end_date)
        engineer    = FeatureEngineer()
        combined_df = engineer.combine_all_features(
            stock_df,
            sentiment.get('news',   pd.DataFrame()),
            sentiment.get('trends', pd.DataFrame()),
            sentiment.get('reddit', pd.DataFrame()),
        )

        # Align to general model's feature columns
        feature_columns = general_model.feature_columns
        for col in feature_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
        engineer.feature_columns = feature_columns

        split_idx        = int(len(combined_df) * (1 - config.TEST_SIZE))
        test_df          = combined_df.iloc[split_idx:].copy().reset_index(drop=True)
        test_returns     = general_model.predict(test_df[feature_columns])
        test_predictions = test_df['close'].values * (1 + test_returns)
        all_returns      = general_model.predict(combined_df[feature_columns])
        all_predictions  = combined_df['close'].values * (1 + all_returns)
        latest_return    = general_model.predict(combined_df[feature_columns].iloc[-1:])[0]
        current_price    = combined_df['close'].iloc[-1]

        combined_df.to_csv(get_feature_path(stock_name), index=False)

        return (current_price * (1 + latest_return), current_price,
                stock_df, combined_df, test_df, test_predictions,
                all_predictions, general_model, engineer, "general_model")

    # ── Per-stock fallback ────────────────────────────────────────────
    model_path   = get_model_path(stock_name)
    feature_path = get_feature_path(stock_name)
    predictor    = StockPredictor()
    engineer     = FeatureEngineer()

    if os.path.exists(model_path) and os.path.exists(feature_path) and not force_retrain:
        predictor.load_model(model_path)
        combined_df = pd.read_csv(feature_path)
        combined_df["Date"] = pd.to_datetime(combined_df["Date"])
        exclude = ['Date', 'target', 'date', 'close', 'open', 'high', 'low', 'volume']
        engineer.feature_columns = [c for c in combined_df.columns if c not in exclude]
        _, X_test, _, _, test_df = engineer.prepare_train_test_split(
            combined_df, test_size=config.TEST_SIZE)
        test_returns     = predictor.predict(X_test)
        test_predictions = test_df['close'].values * (1 + test_returns)
        all_returns      = predictor.predict(combined_df[engineer.feature_columns])
        all_predictions  = combined_df['close'].values * (1 + all_returns)
        latest_return    = predictor.predict(combined_df[engineer.feature_columns].iloc[-1:])[0]
        current_price    = combined_df["close"].iloc[-1]
        return (current_price * (1 + latest_return), current_price,
                combined_df.copy(), combined_df, test_df, test_predictions,
                all_predictions, predictor, engineer, "cached")

    # Fresh per-stock training
    stock_scraper = StockScraper(stock_name)
    stock_df      = stock_scraper.fetch_historical_data(start_date, end_date)
    stock_df      = stock_scraper.calculate_technical_indicators(stock_df)
    sentiment     = _fetch_sentiment(stock_name, start_date, end_date)
    combined_df   = engineer.combine_all_features(
        stock_df,
        sentiment.get('news',   pd.DataFrame()),
        sentiment.get('trends', pd.DataFrame()),
        sentiment.get('reddit', pd.DataFrame()),
    )
    combined_df.to_csv(feature_path, index=False)
    X_train, X_test, y_train, y_test, test_df = engineer.prepare_train_test_split(
        combined_df, test_size=config.TEST_SIZE)
    predictor.train(X_train, y_train)
    predictor.save_model(model_path)
    test_returns     = predictor.predict(X_test)
    test_predictions = test_df['close'].values * (1 + test_returns)
    all_returns      = predictor.predict(combined_df[engineer.feature_columns])
    all_predictions  = combined_df['close'].values * (1 + all_returns)
    latest_return    = predictor.predict(combined_df[engineer.feature_columns].iloc[-1:])[0]
    current_price    = stock_df['close'].iloc[-1]
    return (current_price * (1 + latest_return), current_price,
            stock_df, combined_df, test_df, test_predictions,
            all_predictions, predictor, engineer, "trained")


# -------------------- Martingale Display Helper --------------------
def display_martingale_section(ticker, combined_df, predictor, engineer,
                                mg_capital, mg_base_bet, mg_max_mult,
                                mg_threshold, mg_cost):
    """Render the full Martingale backtest panel for one stock."""

    st.markdown("#### 🎲 Martingale Strategy Backtest")
    st.warning(
        "⚠️ Martingale doubles bet size after each loss and resets on a win. "
        "High-risk strategy — for analysis and research only."
    )

    mg = MartingaleBacktester(
        initial_capital=mg_capital,
        base_bet_pct=mg_base_bet,
        max_multiplier=mg_max_mult,
        transaction_cost=mg_cost,
        signal_threshold=mg_threshold,
    )

    try:
        mg_metrics, mg_portfolio, mg_trades = mg.run(
            combined_df=combined_df,
            predictor=predictor,
            feature_columns=engineer.feature_columns,
            test_size=config.TEST_SIZE,
        )
    except Exception as e:
        st.error(f"❌ Martingale backtest failed for {ticker}: {e}")
        return

    # ── Key metrics row ─────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        ret_val  = mg_metrics["Strategy_Return_%"]
        ret_sign = "+" if ret_val >= 0 else ""
        st.metric(
            "Final Capital",
            f"${mg_metrics['Final_Capital']:,.2f}",
            f"{ret_sign}{ret_val:.2f}%"
        )
    with m2:
        st.metric("Win Rate", f"{mg_metrics['Win_Rate_%']:.1f}%")
    with m3:
        st.metric("Sharpe Ratio", f"{mg_metrics['Sharpe_Ratio']:.3f}")
    with m4:
        st.metric("Max Drawdown", f"{mg_metrics['Max_Drawdown_%']:.1f}%")
    with m5:
        st.metric(
            "Max Loss Streak",
            str(mg_metrics["Max_Loss_Streak"]),
            help="Consecutive losing trades — triggers bet doubling"
        )

    # ── Secondary metrics row ────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Total Trades", str(mg_metrics["Total_Trades"]))
    with s2:
        st.metric("Profit Factor", f"{mg_metrics['Profit_Factor']:.3f}")
    with s3:
        st.metric(
            "Max Multiplier Reached",
            f"{mg_metrics['Max_Multiplier_Used']}×",
            help=f"Cap set at {mg_max_mult}×"
        )

    # ── Interpretation ───────────────────────────────────────────────
    ret_val = mg_metrics["Strategy_Return_%"]
    if ret_val > 0:
        st.success(f"✅ Martingale ended profitably at {ret_val:+.2f}% return.")
    else:
        st.error(f"❌ Martingale ended in a loss at {ret_val:+.2f}% return.")

    if mg_metrics["Max_Loss_Streak"] >= 3:
        st.warning(
            f"⚠️ Hit {mg_metrics['Max_Loss_Streak']} consecutive losses — "
            f"bet multiplier climbed to "
            f"{min(2**(mg_metrics['Max_Loss_Streak']-1), mg_max_mult)}× at peak."
        )

    if mg_metrics["Max_Drawdown_%"] > 20:
        st.error(
            f"🚨 Drawdown of {mg_metrics['Max_Drawdown_%']:.1f}% — "
            "Martingale amplified losses significantly during this period."
        )

    # ── Portfolio equity curve chart ─────────────────────────────────
    if not mg_portfolio.empty:
        fig_mg = go.Figure()

        fig_mg.add_trace(go.Scatter(
            x=mg_portfolio["Date"],
            y=mg_portfolio["Capital"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#9C27B0", width=2),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.08)"
        ))

        # Horizontal line at starting capital
        fig_mg.add_hline(
            y=mg_capital,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            annotation_text=f"Start: ${mg_capital:,.0f}",
            annotation_position="top left",
            annotation_font_color="gray"
        )

        fig_mg.update_layout(
            title=f"{ticker} – Martingale Portfolio Equity Curve",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                title="Date", showgrid=True, gridcolor="#f0f0f0",
                linecolor="#e0e0e0", tickfont=dict(color="#1a1a1a")
            ),
            yaxis=dict(
                title="Portfolio Value ($)", showgrid=True,
                gridcolor="#f0f0f0", linecolor="#e0e0e0",
                tickfont=dict(color="#1a1a1a")
            ),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            font=dict(family="Arial, sans-serif", size=12, color="#333")
        )
        st.plotly_chart(fig_mg, use_container_width=True)

    # ── Trade log table ──────────────────────────────────────────────
    # Using st.checkbox instead of st.expander to avoid nested-expander errors
    if not mg_trades.empty:
        if st.checkbox(f"📋 Show Trade Log — last 30 trades ({ticker})",
                       key=f"mg_tradelog_{ticker}", value=False):
            display_trades = mg_trades.copy()

            display_trades["Result"] = display_trades["Won"].apply(
                lambda w: "✅ Win" if w else "❌ Loss"
            )
            display_trades["Bet_Pct"] = display_trades["Bet_Pct"].apply(
                lambda x: f"{x:.2f}%"
            )
            display_trades["PnL"] = display_trades["PnL"].apply(
                lambda x: f"+${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
            )
            display_trades["Capital_After"] = display_trades["Capital_After"].apply(
                lambda x: f"${x:,.2f}"
            )

            cols_to_show = ["Date", "Signal", "Multiplier", "Bet_Pct",
                            "PnL", "Capital_After", "Result"]
            existing_cols = [c for c in cols_to_show if c in display_trades.columns]

            st.dataframe(
                display_trades[existing_cols].tail(30),
                use_container_width=True,
                hide_index=True
            )

    # ── Settings recap ───────────────────────────────────────────────
    if st.checkbox(f"⚙️ Show Martingale Settings Used ({ticker})",
                   key=f"mg_settings_{ticker}", value=False):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Base Bet", f"{mg_base_bet*100:.1f}%")
        sc2.metric("Max Cap", f"{mg_max_mult}×")
        sc3.metric("Threshold", f"{mg_threshold*100:.2f}%")
        sc4.metric("Tx Cost", f"{mg_cost*100:.2f}%")


# -------------------- Main Stock Display --------------------
def display_stock_prediction(ticker, prediction, current_price, stock_df,
                              combined_df, test_df, test_predictions,
                              all_predictions, snapshot, predictor, engineer,
                              sl_config=None, mg_enabled=False,
                              mg_capital=10000, mg_base_bet=0.05,
                              mg_max_mult=8, mg_threshold=0.005, mg_cost=0.001):
    """Display prediction results for a single stock."""

    delta     = prediction - current_price
    delta_pct = (delta / current_price) * 100

    # Stock header
    st.markdown(f"<div class='stock-header'>📊 {ticker}</div>", unsafe_allow_html=True)

    # ── Main Prediction Cards ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=None)

    with col2:
        st.metric(
            label="Predicted Price (Next Day)",
            value=f"${prediction:.2f}",
            delta=f"{delta:.2f} ({delta_pct:.2f}%)"
        )

    with col3:
        signal       = "BUY 📈" if delta > 0 else "SELL 📉"
        signal_class = "signal-buy" if delta > 0 else "signal-sell"
        st.markdown("<p style='color: #1a1a1a; font-weight: bold;'>Trading Signal</p>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='signal-badge {signal_class}'>{signal}</div>",
                    unsafe_allow_html=True)
        st.caption("Based on ML prediction")

    # ── Stop Loss Panel ──────────────────────────────────────────────
    if sl_config is not None:
        st.markdown("#### 🛡️ Stop Loss Analysis")

        atr_series = compute_atr(
            combined_df["high"], combined_df["low"], combined_df["close"],
            sl_config.atr_window
        )
        latest_atr = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else None

        sl_mgr = StopLossManager(sl_config)
        sl_mgr.open_position(current_price, entry_atr=latest_atr)
        snap = sl_mgr.snapshot()

        pred_triggered, _, _ = sl_mgr.check(prediction)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Mode", snap["mode"])
        with sc2:
            st.metric("Stop Level", f"${snap['stop_price']:.2f}",
                      delta=f"{snap['pct_from_entry']:.2f}%")
        with sc3:
            st.metric("ATR (14d)", f"${latest_atr:.2f}" if latest_atr else "N/A")
        with sc4:
            if snap.get("peak_price"):
                st.metric("Peak / Ref", f"${snap['peak_price']:.2f}")

        if pred_triggered:
            st.error(
                f"⚠️ **STOP LOSS OVERRIDE** — "
                f"Predicted price ${prediction:.2f} is BELOW stop level "
                f"${snap['stop_price']:.2f}. Signal changed to **SELL**."
            )
            st.markdown(
                "<div class='signal-badge signal-sell' "
                "style='font-size:14px;padding:6px 16px'>"
                "SELL (Stop Loss) 🛑</div>",
                unsafe_allow_html=True
            )
        else:
            gap_pct = (prediction - snap["stop_price"]) / snap["stop_price"] * 100
            st.success(
                f"✅ Predicted price ${prediction:.2f} is "
                f"**{gap_pct:.2f}% above** the stop level. ML signal stands."
            )

        st.session_state[f"stop_price_{ticker}"] = snap["stop_price"]

    # ── Market Snapshot ──────────────────────────────────────────────
    with st.expander(f"📊 Market Snapshot - {ticker}", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Open", f"${snapshot['open']:.2f}" if snapshot['open'] else "N/A")
        with c2:
            st.metric("Previous Close",
                      f"${snapshot['previous_close']:.2f}" if snapshot['previous_close'] else "N/A")
        with c3:
            st.metric("Day High", f"${snapshot['day_high']:.2f}" if snapshot['day_high'] else "N/A")
        with c4:
            st.metric("Day Low", f"${snapshot['day_low']:.2f}" if snapshot['day_low'] else "N/A")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("52W High", f"${snapshot['year_high']:.2f}" if snapshot['year_high'] else "N/A")
        with c6:
            st.metric("52W Low", f"${snapshot['year_low']:.2f}" if snapshot['year_low'] else "N/A")
        with c7:
            st.metric("Volume", f"{snapshot['volume']:,}" if snapshot['volume'] else "N/A")
        with c8:
            st.metric("Avg Volume", f"{snapshot['avg_volume']:,}" if snapshot['avg_volume'] else "N/A")

    # ── Price Chart ──────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df['Date'], y=combined_df['close'],
        mode='lines', name='Actual Price',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.1)'
    ))

    valid_predictions = all_predictions[~np.isnan(all_predictions)] if len(all_predictions) > 0 else []
    if len(valid_predictions) > 0:
        fig.add_trace(go.Scatter(
            x=combined_df['Date'], y=all_predictions,
            mode='lines',
            name=f'Model Predictions (std={np.std(all_predictions):.2f})',
            line=dict(color='#FF9800', width=2, dash='dash')
        ))
    else:
        st.warning(f"⚠️ No valid predictions generated for {ticker}")

    next_day = stock_df['Date'].iloc[-1] + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[stock_df['Date'].iloc[-1], next_day],
        y=[current_price, prediction],
        mode='lines+markers', name='Next Day Forecast',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10, color='#4CAF50')
    ))

    # Add stop loss line if available
    stop_px = st.session_state.get(f"stop_price_{ticker}")
    if stop_px:
        fig.add_hline(
            y=stop_px, line_dash="dot", line_color="red", line_width=2,
            annotation_text=f"Stop Loss: ${stop_px:.2f}",
            annotation_position="bottom right",
            annotation_font_color="red"
        )
        below_stop = combined_df[combined_df["close"] < stop_px]
        if not below_stop.empty:
            fig.add_trace(go.Scatter(
                x=below_stop["Date"], y=below_stop["close"],
                mode="markers", name="Below Stop",
                marker=dict(color="red", size=6, symbol="x"),
            ))

    fig.update_layout(
        title=f"{ticker} Price Analysis",
        height=400,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(title="Date", title_font=dict(color='#1a1a1a'),
                   tickfont=dict(color='#1a1a1a'), showgrid=True,
                   gridcolor='#f0f0f0', linecolor='#e0e0e0'),
        yaxis=dict(title="Price ($)", title_font=dict(color='#1a1a1a'),
                   tickfont=dict(color='#1a1a1a'), showgrid=True,
                   gridcolor='#f0f0f0', linecolor='#e0e0e0'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        font=dict(family="Arial, sans-serif", size=12, color="#333")
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Martingale Backtest Panel ────────────────────────────────────
    # NOTE: Cannot use st.expander here because display_stock_prediction is
    # already called inside an outer context that may be an expander.
    # We render the section directly, gated by a divider and header instead.
    if mg_enabled:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #7b1fa2 0%, #9c27b0 100%);
                    color: white; padding: 12px 20px; border-radius: 10px;
                    margin: 20px 0 10px 0; font-size: 18px; font-weight: 700;'>
            🎲 Martingale Strategy Backtest
        </div>
        """, unsafe_allow_html=True)
        display_martingale_section(
            ticker=ticker,
            combined_df=combined_df,
            predictor=predictor,
            engineer=engineer,
            mg_capital=mg_capital,
            mg_base_bet=mg_base_bet,
            mg_max_mult=mg_max_mult,
            mg_threshold=mg_threshold,
            mg_cost=mg_cost,
        )

    # ── Debug Info ───────────────────────────────────────────────────
    with st.expander(f"🔍 Debug Info - {ticker}"):
        st.write("**Prediction Statistics:**")
        st.write(f"- Number of predictions: {len(all_predictions)}")
        st.write(f"- Min: ${np.min(all_predictions):.2f}")
        st.write(f"- Max: ${np.max(all_predictions):.2f}")
        st.write(f"- Mean: ${np.mean(all_predictions):.2f}")
        st.write(f"- Std Dev: ${np.std(all_predictions):.2f}")
        if np.std(all_predictions) < 1.0:
            st.error("⚠️ Predictions have very low variance - model may not be learning!")
        st.write("\n**Current vs Predicted:**")
        st.write(f"- Current price: ${current_price:.2f}")
        st.write(f"- Next day prediction: ${prediction:.2f}")
        st.write(
            f"- Change: ${prediction - current_price:.2f} "
            f"({((prediction - current_price) / current_price * 100):.2f}%)"
        )

    st.markdown("---")


# -------------------- UI Output --------------------
if train_button:
    stock_tickers = [t.strip().upper() for t in stock_input.split(",") if t.strip()]

    if len(stock_tickers) < 3 or len(stock_tickers) > 4:
        st.error("⚠️ Please enter 3-4 stock tickers separated by commas.")
    else:
        st.markdown(
            f"<p style='color: #1a1a1a; background-color: #e3f2fd; padding: 12px; "
            f"border-radius: 8px; border-left: 4px solid #2196F3;'>"
            f"🔄 Processing {len(stock_tickers)} stocks: {', '.join(stock_tickers)}</p>",
            unsafe_allow_html=True
        )

        all_results = {}

        for idx, ticker in enumerate(stock_tickers):
            st.markdown(f"### Processing {ticker} ({idx+1}/{len(stock_tickers)})...")

            with st.spinner(f"🔄 Generating prediction for {ticker}..."):
                try:
                    result = run_pipeline(
                        ticker, start_date, end_date,
                        force_retrain=force_retrain
                    )

                    (prediction, current_price, stock_df, combined_df, test_df,
                     test_predictions, all_predictions, predictor,
                     engineer, status) = result

                    try:
                        snapshot = fetch_quote_snapshot(ticker)
                    except Exception:
                        st.warning(f"⚠️ Could not fetch real-time data for {ticker}.")
                        snapshot = {k: None for k in [
                            "open", "previous_close", "day_high", "day_low",
                            "volume", "avg_volume", "year_high", "year_low"
                        ]}

                    all_results[ticker] = {
                        'prediction':      prediction,
                        'current_price':   current_price,
                        'stock_df':        stock_df,
                        'combined_df':     combined_df,
                        'test_df':         test_df,
                        'test_predictions': test_predictions,
                        'all_predictions': all_predictions,
                        'snapshot':        snapshot,
                        'predictor':       predictor,
                        'engineer':        engineer,
                        'status':          status,
                    }

                    status_emoji = {"cached": "✅", "general_model": "🌍", "trained": "🔄"}.get(status, "🔄")
                    status_text  = {"cached": "Using cached model", "general_model": "Using general model", "trained": "Trained new model"}.get(status, status)
                    st.markdown(
                        f"<p style='color: #1a1a1a; background-color: #e8f5e9; padding: 12px; "
                        f"border-radius: 8px; border-left: 4px solid #4CAF50;'>"
                        f"{status_emoji} <b>{ticker}</b>: {status_text}</p>",
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"❌ Error processing {ticker}: {str(e)}")
                    import traceback
                    with st.expander(f"🔍 Error details for {ticker}"):
                        st.code(traceback.format_exc())

        # ── Summary Comparison ───────────────────────────────────────
        if all_results:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("## 📊 Summary Comparison")

            cols = st.columns(len(all_results))
            for idx, (ticker, data) in enumerate(all_results.items()):
                with cols[idx]:
                    d     = data['prediction'] - data['current_price']
                    d_pct = (d / data['current_price']) * 100
                    st.metric(label=f"{ticker}", value=f"${data['prediction']:.2f}",
                              delta=f"{d_pct:.2f}%")
                    sig   = "BUY 📈" if d > 0 else "SELL 📉"
                    cls   = "signal-buy" if d > 0 else "signal-sell"
                    st.markdown(
                        f"<div class='signal-badge {cls}' "
                        f"style='font-size: 12px; padding: 5px 15px;'>{sig}</div>",
                        unsafe_allow_html=True
                    )

            # ── Martingale Summary Comparison Table ──────────────────
            if mg_enabled and len(all_results) > 1:
                st.markdown("### 🎲 Martingale Strategy Comparison")
                mg_summary_rows = []
                for ticker, data in all_results.items():
                    try:
                        mg = MartingaleBacktester(
                            initial_capital=mg_capital,
                            base_bet_pct=mg_base_bet,
                            max_multiplier=mg_max_mult,
                            transaction_cost=mg_cost,
                            signal_threshold=mg_threshold,
                        )
                        mg_m, _, _ = mg.run(
                            combined_df=data["combined_df"],
                            predictor=data["predictor"],
                            feature_columns=data["engineer"].feature_columns,
                            test_size=config.TEST_SIZE,
                        )
                        mg_summary_rows.append({
                            "Ticker":          ticker,
                            "Final Capital":   f"${mg_m['Final_Capital']:,.2f}",
                            "Return %":        f"{mg_m['Strategy_Return_%']:+.2f}%",
                            "Win Rate":        f"{mg_m['Win_Rate_%']:.1f}%",
                            "Sharpe":          f"{mg_m['Sharpe_Ratio']:.3f}",
                            "Max Drawdown":    f"{mg_m['Max_Drawdown_%']:.1f}%",
                            "Max Loss Streak": str(mg_m["Max_Loss_Streak"]),
                            "Max Multiplier":  f"{mg_m['Max_Multiplier_Used']}×",
                        })
                    except Exception:
                        mg_summary_rows.append({
                            "Ticker": ticker, "Final Capital": "Error",
                            "Return %": "—", "Win Rate": "—",
                            "Sharpe": "—", "Max Drawdown": "—",
                            "Max Loss Streak": "—", "Max Multiplier": "—",
                        })

                if mg_summary_rows:
                    st.dataframe(
                        pd.DataFrame(mg_summary_rows).set_index("Ticker"),
                        use_container_width=True
                    )

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("## 📈 Detailed Predictions")

            for ticker, data in all_results.items():
                display_stock_prediction(
                    ticker=ticker,
                    prediction=data['prediction'],
                    current_price=data['current_price'],
                    stock_df=data['stock_df'],
                    combined_df=data['combined_df'],
                    test_df=data['test_df'],
                    test_predictions=data['test_predictions'],
                    all_predictions=data['all_predictions'],
                    snapshot=data['snapshot'],
                    predictor=data['predictor'],
                    engineer=data['engineer'],
                    sl_config=sl_config,
                    mg_enabled=mg_enabled,
                    mg_capital=mg_capital,
                    mg_base_bet=mg_base_bet,
                    mg_max_mult=mg_max_mult,
                    mg_threshold=mg_threshold,
                    mg_cost=mg_cost,
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
                <p><b>Strategies:</b> Stop Loss (Fixed / Trailing / ATR) + Martingale Backtesting</p>
                </div>
                """, unsafe_allow_html=True)

else:
    # ── Welcome Screen ───────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 40px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px; text-align: center; margin: 50px 0;'>
        <h2 style='color: #1a1a1a; margin-bottom: 20px;'>
            Welcome to StockSenseAI - Multi-Stock Edition
        </h2>
        <p style='color: #2c3e50; font-size: 18px;'>
            Enter 3-4 stock tickers in the sidebar and click <b>Generate Predictions</b> to get started.
        </p>
        <p style='color: #34495e; margin-top: 20px;'>
            Our AI analyzes market data, news sentiment, and social trends to predict next-day
            stock prices — with built-in Stop Loss and Martingale strategy analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;
                    border-left: 4px solid #2196F3;'>
            <h3 style='color: #2196F3;'>📊 Multi-Stock Analysis</h3>
            <p style='color: #2c3e50;'>Analyze and predict 3-4 stocks simultaneously
            with independent models</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;
                    border-left: 4px solid #FF9800;'>
            <h3 style='color: #FF9800;'>🤖 Smart Caching</h3>
            <p style='color: #2c3e50;'>Per-stock model caching for faster
            subsequent predictions</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;
                    border-left: 4px solid #4CAF50;'>
            <h3 style='color: #4CAF50;'>🛡️ Stop Loss</h3>
            <p style='color: #2c3e50;'>Fixed, Trailing, and ATR-based stop loss
            modes with chart overlay</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px;
                    border-left: 4px solid #9C27B0;'>
            <h3 style='color: #9C27B0;'>🎲 Martingale</h3>
            <p style='color: #2c3e50;'>Martingale strategy backtest with equity
            curve, trade log, and risk metrics</p>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class='footer'>
    <p><b>StockSenseAI</b> | Multi-Stock Edition | Stop Loss | Martingale Strategy |
    Train-on-Demand | Per-Stock Model Caching</p>
</div>
""", unsafe_allow_html=True)