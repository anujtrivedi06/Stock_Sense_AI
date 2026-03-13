# data_scrapers/reddit_scraper.py
"""
Reddit sentiment scraper using free JSON endpoints
Adds DATE stamps for temporal alignment
Subreddits are scraped in parallel for speed.
"""

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        Returns DAILY aggregated Reddit sentiment.
        All 4 subreddits are fetched in parallel — ~4x faster than serial.
        A small per-thread delay is kept to avoid hammering Reddit's API.
        """
        all_posts = []
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockstobuytoday']

        def fetch_with_delay(subreddit):
            # Small jitter so threads don't all hit Reddit at the exact same ms
            time.sleep(0.5)
            return self.scrape_reddit_simple(ticker, subreddit)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_with_delay, sr): sr for sr in subreddits}
            for future in as_completed(futures):
                try:
                    posts = future.result()
                    if not posts.empty:
                        all_posts.append(posts)
                except Exception as e:
                    print(f"✗ Reddit thread error ({futures[future]}): {e}")

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