# data_scrapers/trends_scraper.py
"""
Google Trends scraper using pytrends
"""
from pytrends.request import TrendReq
import pandas as pd
import time

class TrendsScraper:
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)
    
    def get_search_trends(self, keyword, start_date, end_date):
        """
        Get Google Trends data for a keyword
        """
        print(f"Fetching Google Trends for '{keyword}'...")
        
        try:
            # Build payload
            self.pytrends.build_payload(
                [keyword],
                timeframe=f'{start_date} {end_date}'
            )
            
            # Get interest over time
            trends_df = self.pytrends.interest_over_time()
            
            if not trends_df.empty:
                trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')
                trends_df.reset_index(inplace=True)
                trends_df.columns = ['Date', 'search_volume']
                
                # Normalize to 0-1 scale
                max_vol = trends_df['search_volume'].max()
                if max_vol > 0:
                    trends_df['search_interest'] = trends_df['search_volume'] / max_vol
                else:
                    trends_df['search_interest'] = 0
                
                print(f"✓ Fetched {len(trends_df)} days of Google Trends data")
                return trends_df
            else:
                print("✗ No Google Trends data available")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keyword):
        """
        Get related queries and topics
        """
        try:
            self.pytrends.build_payload([keyword])
            related = self.pytrends.related_queries()
            
            if related and keyword in related:
                rising = related[keyword]['rising']
                if rising is not None and not rising.empty:
                    return rising['query'].tolist()[:5]
            
            return []
            
        except Exception as e:
            print(f"✗ Error fetching related queries: {e}")
            return []