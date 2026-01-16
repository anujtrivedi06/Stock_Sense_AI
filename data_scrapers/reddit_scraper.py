# data_scrapers/reddit_scraper.py
"""
Reddit sentiment scraper (using free PRAW API)
Fallback to web scraping if API not available
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

class RedditScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_reddit_simple(self, ticker, subreddit='wallstreetbets'):
        """
        Simple Reddit scraping without API (fallback method)
        Scrapes from Reddit's JSON endpoint
        """
        url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&restrict_sr=1&limit=50"
        posts_data = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(2)  # Be respectful to Reddit's servers
            
            if response.status_code == 200:
                data = response.json()
                posts = data['data']['children']
                
                for post in posts:
                    try:
                        post_data = post['data']
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        score = post_data.get('score', 0)
                        
                        # Combine title and text
                        full_text = f"{title} {selftext}"
                        
                        # Get sentiment
                        sentiment = self.analyzer.polarity_scores(full_text)
                        
                        posts_data.append({
                            'text': title,
                            'score': score,
                            'sentiment_score': sentiment['compound'],
                            'positive': sentiment['pos'],
                            'negative': sentiment['neg']
                        })
                    except:
                        continue
                
                print(f"✓ Scraped {len(posts_data)} Reddit posts from r/{subreddit}")
            else:
                print(f"✗ Reddit returned status code: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error scraping Reddit: {e}")
        
        return pd.DataFrame(posts_data)
    
    def get_reddit_sentiment(self, ticker):
        """
        Get aggregated Reddit sentiment
        """
        all_posts = []
        
        # Scrape multiple subreddits
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        
        for subreddit in subreddits:
            posts = self.scrape_reddit_simple(ticker, subreddit)
            if not posts.empty:
                all_posts.append(posts)
            time.sleep(2)  # Rate limiting
        
        if all_posts:
            combined = pd.concat(all_posts, ignore_index=True)
            
            # Weight sentiment by post score (engagement)
            if combined['score'].sum() > 0:
                weighted_sentiment = (combined['sentiment_score'] * combined['score']).sum() / combined['score'].sum()
            else:
                weighted_sentiment = combined['sentiment_score'].mean()
            
            sentiment_metrics = {
                'reddit_avg_sentiment': combined['sentiment_score'].mean(),
                'reddit_weighted_sentiment': weighted_sentiment,
                'reddit_volume': len(combined),
                'reddit_engagement': combined['score'].sum(),
                'reddit_positive_ratio': (combined['sentiment_score'] > 0.05).sum() / len(combined)
            }
            
            return sentiment_metrics
        
        return {
            'reddit_avg_sentiment': 0,
            'reddit_weighted_sentiment': 0,
            'reddit_volume': 0,
            'reddit_engagement': 0,
            'reddit_positive_ratio': 0
        }