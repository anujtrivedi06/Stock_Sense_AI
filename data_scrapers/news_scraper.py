import config
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf


class NewsScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            # A real browser UA reduces FinViz block rate
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            )
        }

    # ------------------------------------------------------------------ #
    #  SOURCE 1 — FinViz                                                   #
    # ------------------------------------------------------------------ #
    def scrape_finviz_news(self, ticker):
        """
        Scrape FinViz headlines with proper date carry-forward.
        Bumped to 500 rows (was 200) to capture ~3-4 weeks of history.
        Added a hard fallback so a bad date row doesn't reset current_date
        to None and corrupt subsequent rows.
        """
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        news_data = []

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            news_table = soup.find(id='news-table')
            if not news_table:
                print(f"✗ FinViz: news table not found (possible block or ticker error) for {ticker}")
                return pd.DataFrame()

            rows = news_table.findAll('tr')
            current_date = None

            for row in rows[:500]:
                try:
                    headline = row.a.get_text(strip=True)
                    time_info = row.td.text.strip()

                    parts = time_info.split()
                    if len(parts) == 2:
                        # Format: "Jul-01-24 09:15AM"
                        try:
                            current_date = datetime.strptime(parts[0], "%b-%d-%y").date()
                        except ValueError:
                            pass  # keep previous current_date rather than setting None
                    # else: time-only row, reuse current_date as-is

                    if current_date is None:
                        continue  # skip rows before we've seen any date

                    sentiment = self.analyzer.polarity_scores(headline)
                    news_data.append({
                        'Date':            current_date,
                        'headline':        headline,
                        'sentiment_score': sentiment['compound'],
                        'positive':        sentiment['pos'],
                        'negative':        sentiment['neg'],
                        'neutral':         sentiment['neu'],
                    })

                except Exception:
                    continue

            print(f"✓ FinViz: {len(news_data)} headlines for {ticker}")

        except Exception as e:
            print(f"✗ FinViz error for {ticker}: {e}")

        return pd.DataFrame(news_data)

    # ------------------------------------------------------------------ #
    #  SOURCE 2 — NewsAPI  (replaces broken Yahoo scraper)                #
    # ------------------------------------------------------------------ #
    def scrape_newsapi(self, ticker):
        """
        NewsAPI free tier: up to 100 articles per request, 30-day window.
        We now page through 3 pages (300 articles) and also search the
        company alias so we don't miss articles that use the full name.
        Falls back gracefully if the API key is missing/invalid.
        """
        api_key = getattr(config, 'NEWS_API_KEY', None)
        if not api_key:
            print("✗ NewsAPI: no API key in config")
            return pd.DataFrame()

        # alias   = COMPANY_ALIASES.get(ticker.upper(), ticker)
        # Search both ticker symbol AND company name for better coverage
        # Ticker → full company name used as a fallback search term
        stock_details = yf.Ticker(ticker)
        info = stock_details.info
        company_name = (
            info.get('displayName') or
            info.get('shortName') or
            info.get('longName') or
            ticker
        )
        query   = f"{ticker} OR {company_name}" if company_name != ticker else ticker
        url     = "https://newsapi.org/v2/everything"
        news_data = []

        for page in range(1, 4):          # pages 1, 2, 3 → up to 300 articles
            params = {
                "q":        query,
                "sortBy":   "publishedAt",
                "language": "en",
                "apiKey":   api_key,
                "pageSize": 100,
                "page":     page,
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                data = resp.json()

                articles = data.get("articles", [])
                if not articles:
                    break   # no more pages

                for article in articles:
                    title = (article.get("title") or "").strip()
                    if not title or title == "[Removed]":
                        continue
                    pub_date = article.get("publishedAt", "")
                    try:
                        date = pd.to_datetime(pub_date).date()
                    except Exception:
                        continue

                    sentiment = self.analyzer.polarity_scores(title)
                    news_data.append({
                        "Date":            date,
                        "headline":        title,
                        "sentiment_score": sentiment["compound"],
                        "positive":        sentiment["pos"],
                        "negative":        sentiment["neg"],
                        "neutral":         sentiment["neu"],
                    })

            except Exception as e:
                print(f"✗ NewsAPI page {page} error: {e}")
                break

        print(f"✓ NewsAPI: {len(news_data)} articles ({query})")
        return pd.DataFrame(news_data)

    # ------------------------------------------------------------------ #
    #  SOURCE 3 — AlphaVantage News  (replaces broken Yahoo scraper)      #
    # ------------------------------------------------------------------ #
    def scrape_alphavantage_news(self, ticker):
        """
        AlphaVantage News & Sentiment endpoint (free tier, 25 req/day).
        Returns up to 1000 articles with proper timestamps and topics.

        Free key: https://www.alphavantage.co/support/#api-key
        Add AV_API_KEY to your config.py
        """
        api_key = getattr(config, 'AV_API_KEY', None)
        if not api_key:
            print("✗ AlphaVantage: no AV_API_KEY in config — skipping")
            return pd.DataFrame()

        url = "https://www.alphavantage.co/query"
        params = {
            "function":  "NEWS_SENTIMENT",
            "tickers":   ticker,
            "limit":     1000,
            "apikey":    api_key,
        }

        news_data = []
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

            for item in data.get("feed", []):
                time_published = item.get("time_published", "")
                # Format: "20240101T093000"
                try:
                    date = datetime.strptime(time_published[:8], "%Y%m%d").date()
                except Exception:
                    continue

                title = (item.get("title") or "").strip()
                if not title:
                    continue

                # AlphaVantage gives us a pre-computed relevance score per ticker
                ticker_sentiment = 0.0
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() == ticker.upper():
                        try:
                            ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                        except Exception:
                            pass
                        break

                # Use VADER on title as well, then average with AV score
                vader = self.analyzer.polarity_scores(title)
                blended = (vader['compound'] + ticker_sentiment) / 2

                news_data.append({
                    "Date":            date,
                    "headline":        title,
                    "sentiment_score": blended,
                    "positive":        vader['pos'],
                    "negative":        vader['neg'],
                    "neutral":         vader['neu'],
                })

            print(f"✓ AlphaVantage: {len(news_data)} articles")

        except Exception as e:
            print(f"✗ AlphaVantage error: {e}")

        return pd.DataFrame(news_data)

    # ------------------------------------------------------------------ #
    #  SOURCE 4 — Moneycontrol News  (For Indian Stocks)      #
    # ------------------------------------------------------------------ #
    def scrape_moneycontrol(self, ticker):
        """
        Moneycontrol is the most widely read Indian financial news source.
        Strip .NS or .BO suffix before searching.
        """
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        url = f"https://www.moneycontrol.com/stocks/cmsedition/rss/{clean_ticker}_news.xml"
        news_data = []
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            for item in soup.find_all('item')[:100]:
                title = item.find('title')
                pub_date = item.find('pubDate')
                if not title or not pub_date:
                    continue
                try:
                    date = pd.to_datetime(pub_date.get_text()).date()
                except Exception:
                    continue
                sentiment = self.analyzer.polarity_scores(title.get_text())
                news_data.append({
                    'Date': date,
                    'headline': title.get_text(),
                    'sentiment_score': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                })
            print(f"✓ Moneycontrol: {len(news_data)} articles for {ticker}")
        except Exception as e:
            print(f"✗ Moneycontrol error for {ticker}: {e}")
        return pd.DataFrame(news_data)

    # ------------------------------------------------------------------ #
    #  SOURCE 5 — Economic Times News  (For Indian Stocks)      #
    # ------------------------------------------------------------------ #
    def scrape_economic_times(self, ticker):
        """
        Economic Times RSS feed — covers NSE/BSE stocks well.
        """
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        url = f"https://economictimes.indiatimes.com/rssfeedsdefault.cms"
        # ET doesn't have per-ticker RSS so we search their site
        search_url = f"https://economictimes.indiatimes.com/searchresult.cms?query={clean_ticker}"
        news_data = []
        try:
            response = requests.get(search_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('div', class_='eachStory')[:50]
            for article in articles:
                title_tag = article.find('h3') or article.find('h4')
                date_tag  = article.find('time')
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                try:
                    date = pd.to_datetime(date_tag['datetime']).date() if date_tag else datetime.today().date()
                except Exception:
                    date = datetime.today().date()
                sentiment = self.analyzer.polarity_scores(title)
                news_data.append({
                    'Date': date,
                    'headline': title,
                    'sentiment_score': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu'],
                })
            print(f"✓ Economic Times: {len(news_data)} articles for {ticker}")
        except Exception as e:
            print(f"✗ Economic Times error for {ticker}: {e}")
        return pd.DataFrame(news_data)
    
    
    # ------------------------------------------------------------------ #
    #  AGGREGATOR                                                          #
    # ------------------------------------------------------------------ #
    def get_daily_sentiment(self, ticker):
        """
        Fetch all sources in parallel, combine, and return a daily
        aggregated DataFrame.

        New columns vs original:
          - positive_ratio  : fraction of articles with compound > +0.05
          - negative_ratio  : fraction of articles with compound < -0.05
          (feature_engineering.py already referenced these but original
           code never produced them — they were silently zeroed out)
        """
        is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')

        if is_indian:
            tasks = {
                'moneycontrol':   lambda: self.scrape_moneycontrol(ticker),
                'economic_times': lambda: self.scrape_economic_times(ticker),
                'newsapi':        lambda: self.scrape_newsapi(ticker),
            }
        else:
            tasks = {
                'finviz':       lambda: self.scrape_finviz_news(ticker),
                'newsapi':      lambda: self.scrape_newsapi(ticker),
                'alphavantage': lambda: self.scrape_alphavantage_news(ticker),
            }

        all_news = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_news.append(df)
                except Exception as e:
                    print(f"✗ {name} thread error: {e}")

        if not all_news:
            print(f"⚠ No news data retrieved for any source for {ticker}")
            return pd.DataFrame()

        combined = pd.concat(all_news, ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])

        # Drop exact duplicate headlines on the same date (cross-source overlap)
        combined = combined.drop_duplicates(subset=["Date", "headline"])

        daily = combined.groupby("Date").agg(
            avg_sentiment    = ("sentiment_score", "mean"),
            sentiment_std    = ("sentiment_score", "std"),
            news_volume      = ("sentiment_score", "count"),
            positive_ratio   = ("sentiment_score", lambda x: (x > 0.05).mean()),
            negative_ratio   = ("sentiment_score", lambda x: (x < -0.05).mean()),
        ).reset_index()

        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

        print(f"✓ News: {len(daily)} daily rows across "
              f"{(daily['Date'].max() - daily['Date'].min()).days} days")
        return daily