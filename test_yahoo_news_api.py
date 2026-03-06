from data_scrapers.news_scraper2 import NewsScraper
stock_name = "TSLA"
news_scraper = NewsScraper()
news_df = []
news_df = news_scraper.scrape_yahoo_news(stock_name)
# news_df = news_scraper.get_daily_sentiment(stock_name)

print(news_df)