import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'options']


class RedditScraper:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Reddit requires a descriptive User-Agent or returns 429/403
        self.headers = {
            'User-Agent': 'StockSenseAI/1.0 (research bot; sentiment analysis)'
        }

    # ------------------------------------------------------------------ #
    #  PRIMARY: Pushshift / Arctic Shift API (historical, date-sorted)    #
    # ------------------------------------------------------------------ #
    def _scrape_pushshift(self, ticker, subreddit, days_back=90):
        """
        Pushshift indexes Reddit historically and returns results sorted
        by date.
        We use the ArcticShift mirror (api.reddit-search.io) which is
        currently the most reliable free Pushshift-compatible endpoint.
        Falls back to empty DataFrame on any error so the caller can
        try the Reddit search fallback.
        """
        url = "https://api.reddit-search.io/search/submissions"
        after = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())

        params = {
            "q":          ticker,
            "subreddit":  subreddit,
            "after":      after,
            "size":       100,
            "sort":       "desc",
            "sort_type":  "created_utc",
        }

        posts_data = []
        try:
            resp = requests.get(url, params=params,
                                headers=self.headers, timeout=12)
            if resp.status_code != 200:
                return pd.DataFrame()

            items = resp.json().get("data", [])
            for item in items:
                created_utc = item.get("created_utc")
                if not created_utc:
                    continue
                post_date = datetime.utcfromtimestamp(int(created_utc)).date()

                title    = (item.get("title")    or "")
                selftext = (item.get("selftext")  or "")
                score    = int(item.get("score",  0))
                num_comments = int(item.get("num_comments", 0))
                # upvote_ratio is 0-1; values near 0.5 = controversial
                upvote_ratio = float(item.get("upvote_ratio", 1.0))

                full_text = f"{title} {selftext}".strip()
                if not full_text:
                    continue

                sentiment = self.analyzer.polarity_scores(full_text)
                posts_data.append({
                    'Date':              post_date,
                    'sentiment_score':   sentiment['compound'],
                    'positive':          sentiment['pos'],
                    'negative':          sentiment['neg'],
                    'score':             score,
                    'num_comments':      num_comments,
                    'controversy_ratio': 1 - abs(upvote_ratio - 0.5) * 2,
                    # controversy_ratio -> 1 when perfectly split (0.5 ratio),
                    #                   -> 0 when unanimously up/downvoted
                })

        except Exception:
            return pd.DataFrame()

        return pd.DataFrame(posts_data)

    # ------------------------------------------------------------------ #
    #  FALLBACK: Reddit public search.json                                #
    # ------------------------------------------------------------------ #
    def _scrape_reddit_public(self, ticker, subreddit):
        """
        Original Reddit public search endpoint kept as fallback.
        Added sort=new so we at least get recent posts in date order.
        Uses improved User-Agent to reduce 429 errors.
        Bumped limit to 100 (was 50).
        """
        url = (f"https://www.reddit.com/r/{subreddit}/search.json"
               f"?q={ticker}&restrict_sr=1&limit=100&sort=new")
        posts_data = []

        try:
            time.sleep(0.5)   # gentle rate-limit
            resp = requests.get(url, headers=self.headers, timeout=10)

            if resp.status_code != 200:
                return pd.DataFrame()

            posts = resp.json()['data']['children']
            for post in posts:
                d = post['data']
                created_utc = d.get('created_utc')
                if not created_utc:
                    continue
                post_date = datetime.fromtimestamp(created_utc).date()
                title    = d.get('title', '')
                selftext = d.get('selftext', '')
                score    = int(d.get('score', 0))
                num_comments = int(d.get('num_comments', 0))
                upvote_ratio = float(d.get('upvote_ratio', 1.0))

                full_text = f"{title} {selftext}".strip()
                sentiment = self.analyzer.polarity_scores(full_text)

                posts_data.append({
                    'Date':              post_date,
                    'sentiment_score':   sentiment['compound'],
                    'positive':          sentiment['pos'],
                    'negative':          sentiment['neg'],
                    'score':             score,
                    'num_comments':      num_comments,
                    'controversy_ratio': 1 - abs(upvote_ratio - 0.5) * 2,
                })

        except Exception:
            pass

        return pd.DataFrame(posts_data)

    # ------------------------------------------------------------------ #
    #  PER-SUBREDDIT ENTRY POINT                                          #
    # ------------------------------------------------------------------ #
    def scrape_subreddit(self, ticker, subreddit, days_back=90):
        """
        Try Pushshift first; fall back to public Reddit search.
        """
        df = self._scrape_pushshift(ticker, subreddit, days_back=days_back)
        if df.empty:
            df = self._scrape_reddit_public(ticker, subreddit)

        if not df.empty:
            print(f"  r/{subreddit}: {len(df)} posts")
        else:
            print(f"  r/{subreddit}: no data")
        return df

    # ------------------------------------------------------------------ #
    #  AGGREGATOR                                                          #
    # ------------------------------------------------------------------ #
    def get_daily_reddit_sentiment(self, ticker, days_back=90):
        """
        Fetch all subreddits in parallel, combine, aggregate to daily rows.

        Fixed weighted_sentiment: the original lambda referenced the outer
        combined DataFrame via combined.loc[x.index] which broke after
        pd.concat reindexed rows. Now computed cleanly per-group.

        New columns vs original:
          - reddit_comment_volume : total comments (proxy for engagement depth)
          - reddit_controversy    : mean controversy ratio (polarising = signal)
        """
        all_posts = []

        with ThreadPoolExecutor(max_workers=len(SUBREDDITS)) as executor:
            futures = {
                executor.submit(self.scrape_subreddit, ticker, sr, days_back): sr
                for sr in SUBREDDITS
            }
            for future in as_completed(futures):
                sr = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_posts.append(df)
                except Exception as e:
                    print(f"  Reddit thread error ({sr}): {e} for the subreddit stock {ticker}")

        if not all_posts:
            print(f"  No Reddit data retrieved for {ticker}")
            return pd.DataFrame()

        combined = pd.concat(all_posts, ignore_index=True)

        # --- per-group weighted sentiment (bug fix) ---
        def weighted_sent(group):
            scores  = group['sentiment_score'].values
            weights = group['score'].clip(lower=0).values  # downvoted posts = 0 weight
            total_w = weights.sum()
            if total_w == 0:
                return scores.mean()
            return (scores * weights).sum() / total_w

        daily = combined.groupby('Date').apply(
            lambda g: pd.Series({
                'reddit_avg_sentiment':      g['sentiment_score'].mean(),
                'reddit_weighted_sentiment': weighted_sent(g),
                'reddit_volume':             len(g),
                'reddit_engagement':         g['score'].sum(),
                'reddit_comment_volume':     g['num_comments'].sum(),
                'reddit_positive_ratio':     (g['sentiment_score'] > 0.05).mean(),
                'reddit_controversy':        g['controversy_ratio'].mean(),
            })
        ).reset_index()

        print(f"  Reddit: {len(daily)} daily rows, "
              f"{(daily['Date'].max() - daily['Date'].min()).days} day span")
        return daily