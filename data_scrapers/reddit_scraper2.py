# data_scrapers/reddit_scraper.py
"""
Reddit sentiment scraper using free JSON endpoints
Adds DATE stamps for temporal alignment
"""

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time


class RedditScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

    def scrape_reddit_simple(self, ticker, subreddit='wallstreetbets'):
        """
        Scrape Reddit posts with DATE information
        """
        url = f"https://www.reddit.com/r/{subreddit}/search.json?q={ticker}&restrict_sr=1&limit=50"
        posts_data = []

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(2)

            if response.status_code == 200:
                data = response.json()
                posts = data['data']['children']

                for post in posts:
                    try:
                        post_data = post['data']
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        score = post_data.get('score', 0)

                        created_utc = post_data.get('created_utc', None)
                        if created_utc is None:
                            continue

                        post_date = datetime.fromtimestamp(created_utc).date()

                        full_text = f"{title} {selftext}"
                        sentiment = self.analyzer.polarity_scores(full_text)

                        posts_data.append({
                            'Date': post_date,
                            'sentiment_score': sentiment['compound'],
                            'positive': sentiment['pos'],
                            'negative': sentiment['neg'],
                            'score': score
                        })
                    except Exception:
                        continue

                print(f"✓ Scraped {len(posts_data)} posts from r/{subreddit}")

            else:
                print(f"✗ Reddit returned status code {response.status_code}")

        except Exception as e:
            print(f"✗ Error scraping Reddit: {e}")

        return pd.DataFrame(posts_data)

    def get_daily_reddit_sentiment(self, ticker):
        """
        Returns DAILY aggregated Reddit sentiment
        """
        all_posts = []
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockstobuytoday']

        for subreddit in subreddits:
            posts = self.scrape_reddit_simple(ticker, subreddit)
            if not posts.empty:
                all_posts.append(posts)
            time.sleep(2)

        if not all_posts:
            return pd.DataFrame()

        combined = pd.concat(all_posts, ignore_index=True)

        daily_reddit = combined.groupby('Date').agg(
            reddit_avg_sentiment=('sentiment_score', 'mean'),
            reddit_weighted_sentiment=(
                'sentiment_score',
                lambda x: (x * combined.loc[x.index, 'score']).sum() /
                          max(combined.loc[x.index, 'score'].sum(), 1)
            ),
            reddit_volume=('sentiment_score', 'count'),
            reddit_engagement=('score', 'sum'),
            reddit_positive_ratio=('sentiment_score', lambda x: (x > 0.05).mean())
        ).reset_index()

        return daily_reddit
