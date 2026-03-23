from pytrends.request import TrendReq
import pandas as pd
import time
import yfinance as yf

# COMPANY_ALIASES = {
#     'TSLA': 'Tesla',   'AAPL': 'Apple',   'GOOGL': 'Google',
#     'GOOG': 'Google',  'MSFT': 'Microsoft','AMZN': 'Amazon',
#     'META': 'Meta',    'NVDA': 'Nvidia',   'NFLX': 'Netflix',
#     'BABA': 'Alibaba', 'AMD':  'AMD',      'INTC': 'Intel',
#     'UBER': 'Uber',    'LYFT': 'Lyft',     'SNAP': 'Snapchat',
#     'COIN': 'Coinbase','SQ':   'Block',    'SHOP': 'Shopify',
#     'ZM':   'Zoom',
# }


class TrendsScraper:
    def __init__(self):
        # backoff_factor and retries help survive 429s from Google
        self.pytrends = TrendReq(hl='en-US', tz=360)
        # self.pytrends = TrendReq(hl='en-US', tz=360,
        #                          retries=3, backoff_factor=0.5)

    def _fetch_with_retry(self, keyword, timeframe, max_attempts=1):
        """
        Wraps pytrends build_payload + interest_over_time with
        retry/backoff so a single 429 doesn't silently kill the scraper.
        Returns raw trends DataFrame or empty DataFrame on failure.
        """
        # Hardcoded the attempts to 1
        # max_attempts = 1
        for attempt in range(1, max_attempts + 1):
            # HardCoded max attempts to 1
            try:
                self.pytrends.build_payload([keyword], timeframe=timeframe)
                df = self.pytrends.interest_over_time()
                return df
            except Exception as e:
                err = str(e).lower()
                if '429' in err or 'too many' in err:
                    wait = 2 ** attempt   # 2s, 4s, 8s
                    print(f"  Google Trends rate-limited for {keyword}, retrying in {wait}s...")
                    # time.sleep(wait)
                else:
                    print(f"  Google Trends error for {keyword} ({attempt}/{max_attempts}): {e}")
                    if attempt == max_attempts:
                        return pd.DataFrame()
        return pd.DataFrame()

    def get_search_trends(self, keyword, start_date, end_date):
        """
        Fetch daily Google Trends for the keyword over the date range.

        If the window is >270 days pytrends returns weekly data;
        we forward-fill it to daily so the date merge stays aligned.

        New columns:
          - search_interest  : normalised 0-1 (same as original)
          - search_momentum  : 7-day rate of change in search interest
          - search_spike     : 1 if today > 1.5x 30-day rolling mean
        """
        print(f"  Fetching Google Trends for '{keyword}'...")
        timeframe = f'{start_date} {end_date}'

        raw_df = self._fetch_with_retry(keyword, timeframe)

        # If ticker search failed, try company name alias
        if raw_df.empty:
            stock_details = yf.Ticker(keyword)
            info = stock_details.info
            company_name = (
                info.get('displayName') or
                info.get('shortName') or
                info.get('longName') or
                keyword
            )
            if company_name and company_name != keyword:
                print(f"  Trends fallback: trying '{company_name}'...")
                raw_df = self._fetch_with_retry(company_name, timeframe)
                keyword = company_name  # so column rename below works

        if raw_df.empty:
            print(f"  Google Trends: no data for '{keyword}'")
            return pd.DataFrame()

        df = raw_df.drop(columns=['isPartial'], errors='ignore').copy()
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Rename the keyword column regardless of what it was called
        # (handles both ticker and alias cases cleanly)
        kw_col = [c for c in df.columns if c != 'Date']
        if not kw_col:
            return pd.DataFrame()
        df.rename(columns={kw_col[0]: 'search_volume'}, inplace=True)

        # If pytrends returned weekly data (window > ~270 days),
        # resample to daily by forward-filling
        date_diffs = df['Date'].diff().dt.days.dropna()
        is_weekly  = (date_diffs.median() >= 6)

        if is_weekly:
            print(f"  Google Trends returned weekly data for {keyword} — resampling to daily")
            df = (df.set_index('Date')
                    .resample('D')
                    .ffill()
                    .reset_index())

        # Normalise to 0-1
        max_vol = df['search_volume'].max()
        df['search_interest'] = df['search_volume'] / max_vol if max_vol > 0 else 0.0

        # search_momentum: 7-day pct change in interest (leading indicator)
        df['search_momentum'] = df['search_interest'].pct_change(periods=7).fillna(0)

        # search_spike: 1 when today's interest is 1.5x the 30-day rolling mean
        rolling_mean = df['search_interest'].rolling(30, min_periods=5).mean()
        df['search_spike'] = (
            (df['search_interest'] > rolling_mean * 1.5).astype(int)
        )

        print(f"  Google Trends: {len(df)} daily rows for '{keyword}'")
        return df[['Date', 'search_interest', 'search_momentum', 'search_spike']]