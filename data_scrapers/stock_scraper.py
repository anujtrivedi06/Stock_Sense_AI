# data_scrapers/stock_scraper.py
"""
Stock data scraper using yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockScraper:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        
    def fetch_historical_data(self, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance
        Returns DataFrame with OHLCV data
        """
        print(f"Fetching stock data for {self.ticker}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Clean column names
            df.columns = df.columns.str.lower()
            df.rename(columns={'date': 'Date'}, inplace=True)

            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            
            print(f"✓ Fetched {len(df)} days of stock data")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['close'].rolling(window=20).std()
        
        # Price change
        df['Price_Change'] = df['close'].pct_change()
        
        return df