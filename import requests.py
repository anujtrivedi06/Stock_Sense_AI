import requests
import pandas as pd
from datetime import date, datetime
from data_scrapers.news_scraper2 import NewsScraper
from data_scrapers.reddit_scraper2 import RedditScraper
import yfinance as yf
from huggingface_hub import InferenceClient
import os

def scrape_raw_content(ticker: str) -> dict:
    """
    Scrape raw text content from all sources for today.
    Falls back to most recent available if today has no data.
    Returns dict of source -> list of text strings.
    """
    news_scraper = NewsScraper()
    reddit_scraper = RedditScraper()
    today = date.today()
    content = {}

    # ── FinViz ───────────────────────────────────────────────────────
    try:
        df = news_scraper.scrape_finviz_news(ticker)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            today_df = df[df["Date"] == today]
            # fallback to most recent 15 if nothing today
            source_df = today_df if not today_df.empty else df.head(15)
            content["finviz"] = source_df["headline"].dropna().tolist()
    except Exception as e:
        content["finviz"] = []
        print(f"FinViz scrape error: {e}")

    # ── Yahoo Finance ────────────────────────────────────────────────
    try:
        df = news_scraper.scrape_yahoo_news(ticker)
        if not df.empty:
            content["yahoo"] = df["headline"].dropna().tolist()
    except Exception as e:
        content["yahoo"] = []
        print(f"Yahoo scrape error: {e}")

    # ── NewsAPI ──────────────────────────────────────────────────────
    try:
        df = news_scraper.scrape_newsapi(ticker)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            today_df = df[df["Date"] == today]
            source_df = today_df if not today_df.empty else df.head(15)
            content["newsapi"] = source_df["headline"].dropna().tolist()
    except Exception as e:
        content["newsapi"] = []
        print(f"NewsAPI scrape error: {e}")

    # ── Reddit ───────────────────────────────────────────────────────
    # We need raw post text, not just sentiment — so we call the
    # underlying JSON endpoint directly to also capture selftext
    subreddits = ["wallstreetbets", "stocks", "investing", "stockstobuytoday"]
    reddit_posts = []
    headers = {"User-Agent": "StockSentimentBot/1.0 (contact: you@email.com)"}
    
    stock_details = yf.Ticker(ticker)
    company_name = stock_details.info['longName']
    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json?q={company_name}&restrict_sr=1&limit=50&sort=new"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            data = resp.json()
            if "data" not in data:
                print(f"Unexpected Reddit response for r/{sub}: {data}")
                continue
                
            posts = data["data"]["children"]
            for p in posts:
                d = p["data"]
                created = date.fromtimestamp(d.get("created_utc", 0))
                title = d.get("title", "").strip()
                body = d.get("selftext", "").strip()[:500]
                combined = f"{title}. {body}" if body else title
                if combined:
                    reddit_posts.append({
                        "date": created,
                        "text": combined,
                        "score": d.get("score", 0),
                        "subreddit": sub,
                    })
        except Exception as e:
            print(f"Reddit r/{sub} error: {type(e).__name__}: {e}")

    print(f"Total Reddit posts found: {len(reddit_posts)}")
    if reddit_posts:
        df = pd.DataFrame(reddit_posts)
        today_df = df[df["date"] == today]
        # use today's posts if available, else top-scored recent ones
        use_df = today_df if not today_df.empty else df.nlargest(20, "score")
        content["reddit"] = use_df["text"].tolist()

    return content




def summarize_content(content: dict, ticker: str) -> dict:
    """
    Summarize scraped content per source using HuggingFace Inference API.
    Returns dict of source -> summary string.
    """
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN"),
    )


    # client = InferenceClient(api_key="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")
    summaries = {}

    for source, texts in content.items():
        if not texts:
            summaries[source] = "No content available."
            continue

        # Combine texts into a single block
        combined = "\n".join(f"- {t}" for t in texts)
        
        # Truncate to avoid token limits (~2000 chars is safe for most models)
        if len(combined) > 2000:
            combined = combined[:2000] + "..."

        # prompt = {f"""You are a financial analyst. Summarize the following {source} posts/headlines about {ticker} stock briefly. Focus on sentiment, key themes, and any notable events mentioned. Your answer is going to be displayed directly to users, so dont criticise the source much. If it is insufficient just mention it in the summary.

# Content:
# {combined}

# Summary:"""}

        try:
            response = client.summarization(
            combined,
            model="facebook/bart-large-cnn",
        )
            # response = client.chat.completions.create(
            # model="zai-org/GLM-5:novita",
            # messages=[
            #     prompt
            # ],
        # )
            summaries[source] = response["summary_text"]
        except Exception as e:
            print(f"Summarization error for {source}: {e}")
            summaries[source] = "Summarization failed."

    return summaries





# Main
ticker = "DIS"
# print(company_name)
results = scrape_raw_content(ticker)

# print(results)
summaries = summarize_content(results, ticker)

for source, summary in summaries.items():
    print(f"\n=== {source.upper()} ===")
    print(summary)