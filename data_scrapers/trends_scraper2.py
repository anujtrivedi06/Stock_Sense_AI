# data_scrapers/trends_scraper.py
"""
Google Trends scraper using pytrends
Adds DATE normalization for temporal alignment
"""

from pytrends.request import TrendReq
import pandas as pd

class TrendsScraper:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    def get_search_trends(self, keyword, start_date, end_date):
        """
        Get DAILY Google Trends data for a keyword
        """
        print(f"Fetching Google Trends for '{keyword}'...")
        
        try:
            self.pytrends.build_payload(
                [keyword],
                timeframe=f'{start_date} {end_date}'
            )
            
            trends_df = self.pytrends.interest_over_time()
            
            if trends_df.empty:
                print("✗ No Google Trends data available")
                return pd.DataFrame()
            
            trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')
            trends_df.reset_index(inplace=True)
            
            # Ensure consistent Date column
            trends_df.rename(columns={'date': 'Date'}, inplace=True)
            trends_df['Date'] = pd.to_datetime(trends_df['Date']).dt.tz_localize(None)
            
            trends_df.rename(columns={keyword: 'search_volume'}, inplace=True)
            
            # Normalize search interest (0–1)
            max_vol = trends_df['search_volume'].max()
            trends_df['search_interest'] = (
                trends_df['search_volume'] / max_vol if max_vol > 0 else 0
            )
            
            print(f"✓ Fetched {len(trends_df)} days of Google Trends data")
            return trends_df[['Date', 'search_interest']]
                
        except Exception as e:
            print(f"✗ Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keyword):
        """
        Get related rising queries (optional, not used in modeling)
        """
        try:
            self.pytrends.build_payload([keyword])
            related = self.pytrends.related_queries()
            
            if related and keyword in related:
                rising = related[keyword].get('rising')
                if rising is not None and not rising.empty:
                    return rising['query'].tolist()[:5]
            
            return []
            
        except Exception as e:
            print(f"✗ Error fetching related queries: {e}")
            return []
