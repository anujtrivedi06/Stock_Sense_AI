# data_scrapers/news_scraper.py
"""
News sentiment scraper using free sources
Adds DATE stamps for proper temporal alignment
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class NewsScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_finviz_news(self, ticker):
        """
        Scrape news from FinViz with DATE information
        """
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        news_data = []

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            news_table = soup.find(id='news-table')
            if news_table:
                rows = news_table.findAll('tr')

                current_date = None

                for row in rows[:50]:  # recent 50 headlines
                    try:
                        headline = row.a.get_text()
                        time_info = row.td.text.strip()

                        # FinViz format:
                        # Either: "Jul-01-24 09:15AM"
                        # Or:     "09:15AM" (same date as previous row)
                        if len(time_info.split()) == 2:
                            date_str, time_str = time_info.split()
                            current_date = datetime.strptime(date_str, "%b-%d-%y").date()
                        else:
                            time_str = time_info

                        sentiment = self.analyzer.polarity_scores(headline)

                        news_data.append({
                            'Date': current_date,
                            'headline': headline,
                            'sentiment_score': sentiment['compound'],
                            'positive': sentiment['pos'],
                            'negative': sentiment['neg'],
                            'neutral': sentiment['neu']
                        })

                    except Exception:
                        continue

            print(f"✓ Scraped {len(news_data)} news articles from FinViz")

        except Exception as e:
            print(f"✗ Error scraping FinViz: {e}")

        return pd.DataFrame(news_data)

    def scrape_yahoo_news(self, ticker):
        """
        Scrape news from Yahoo Finance
        Yahoo does not expose reliable timestamps,
        so we assign CURRENT DATE
        """
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        news_data = []
        today = datetime.now().date()

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            headlines = soup.find_all('h3', class_='Mb(5px)')

            for headline in headlines[:15]:
                try:
                    text = headline.get_text()
                    sentiment = self.analyzer.polarity_scores(text)

                    news_data.append({
                        'Date': today,
                        'headline': text,
                        'sentiment_score': sentiment['compound'],
                        'positive': sentiment['pos'],
                        'negative': sentiment['neg'],
                        'neutral': sentiment['neu']
                    })
                except Exception:
                    continue

            print(f"✓ Scraped {len(news_data)} news articles from Yahoo Finance")

        except Exception as e:
            print(f"✗ Error scraping Yahoo Finance: {e}")

        return pd.DataFrame(news_data)

    def get_daily_sentiment(self, ticker):
        """
        Returns DAILY aggregated sentiment with Date column
        """
        all_news = []

        finviz_news = self.scrape_finviz_news(ticker)
        if not finviz_news.empty:
            all_news.append(finviz_news)

        yahoo_news = self.scrape_yahoo_news(ticker)
        if not yahoo_news.empty:
            all_news.append(yahoo_news)

        if not all_news:
            return pd.DataFrame()

        combined = pd.concat(all_news, ignore_index=True)
        combined = combined.dropna(subset=['Date'])

        # Aggregate per day
        daily_sentiment = combined.groupby('Date').agg(
            avg_sentiment=('sentiment_score', 'mean'),
            sentiment_std=('sentiment_score', 'std'),
            positive_ratio=('sentiment_score', lambda x: (x > 0.05).mean()),
            negative_ratio=('sentiment_score', lambda x: (x < -0.05).mean()),
            news_volume=('headline', 'count')
        ).reset_index()

        return daily_sentiment
