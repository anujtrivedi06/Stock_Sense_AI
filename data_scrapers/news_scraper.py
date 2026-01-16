# data_scrapers/news_scraper.py
"""
News sentiment scraper using free sources
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

class NewsScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_finviz_news(self, ticker):
        """
        Scrape news from FinViz (free, no API key needed)
        """
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        news_data = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_table = soup.find(id='news-table')
            if news_table:
                rows = news_table.findAll('tr')
                
                for row in rows[:50]:  # Get last 50 news items
                    try:
                        text = row.a.get_text()
                        date_data = row.td.text.split(' ')
                        
                        if len(date_data) == 1:
                            time_str = date_data[0]
                        else:
                            date_str = date_data[0]
                            time_str = date_data[1]
                        
                        # Get sentiment
                        sentiment = self.analyzer.polarity_scores(text)
                        
                        news_data.append({
                            'headline': text,
                            'sentiment_score': sentiment['compound'],
                            'positive': sentiment['pos'],
                            'negative': sentiment['neg'],
                            'neutral': sentiment['neu']
                        })
                    except:
                        continue
            
            print(f"✓ Scraped {len(news_data)} news articles from FinViz")
            
        except Exception as e:
            print(f"✗ Error scraping FinViz: {e}")
        
        return pd.DataFrame(news_data)
    
    def scrape_yahoo_news(self, ticker):
        """
        Scrape news from Yahoo Finance
        """
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        news_data = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news headlines
            headlines = soup.find_all('h3', class_='Mb(5px)')
            
            for headline in headlines[:15]:
                try:
                    text = headline.get_text()
                    sentiment = self.analyzer.polarity_scores(text)
                    
                    news_data.append({
                        'headline': text,
                        'sentiment_score': sentiment['compound'],
                        'positive': sentiment['pos'],
                        'negative': sentiment['neg'],
                        'neutral': sentiment['neu']
                    })
                except:
                    continue
            
            print(f"✓ Scraped {len(news_data)} news articles from Yahoo Finance")
            
        except Exception as e:
            print(f"✗ Error scraping Yahoo Finance: {e}")
        
        return pd.DataFrame(news_data)
    
    def get_aggregated_sentiment(self, ticker):
        """
        Get aggregated sentiment from multiple sources
        """
        all_news = []
        
        # Scrape from multiple sources
        finviz_news = self.scrape_finviz_news(ticker)
        if not finviz_news.empty:
            all_news.append(finviz_news)
        
        yahoo_news = self.scrape_yahoo_news(ticker)
        if not yahoo_news.empty:
            all_news.append(yahoo_news)
        
        if all_news:
            combined = pd.concat(all_news, ignore_index=True)
            
            # Calculate aggregate metrics
            sentiment_metrics = {
                'avg_sentiment': combined['sentiment_score'].mean(),
                'sentiment_std': combined['sentiment_score'].std(),
                'positive_ratio': (combined['sentiment_score'] > 0.05).sum() / len(combined),
                'negative_ratio': (combined['sentiment_score'] < -0.05).sum() / len(combined),
                'news_volume': len(combined)
            }
            
            return sentiment_metrics
        
        return {
            'avg_sentiment': 0,
            'sentiment_std': 0,
            'positive_ratio': 0,
            'negative_ratio': 0,
            'news_volume': 0
        }