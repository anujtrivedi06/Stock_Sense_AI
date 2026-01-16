#!/usr/bin/env python3
"""
Quick test script to verify all dependencies are installed correctly
Run this BEFORE main.py to catch any issues early
"""

import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Core libraries
    imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("yfinance", "yfinance"),
        ("vaderSentiment", "VADER Sentiment"),
        ("requests", "Requests"),
        ("bs4", "BeautifulSoup4"),
        ("pytrends", "Google Trends"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
    ]
    
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            tests_passed += 1
        except ImportError as e:
            print(f"  ❌ {name} - {e}")
            tests_failed += 1
    
    return tests_passed, tests_failed

def test_directories():
    """Test required directory structure"""
    print("\n🔍 Testing directory structure...\n")
    
    import os
    
    required_dirs = [
        "data_scrapers",
        "features",
        "model",
        "dashboard",
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✅ {dir_name}/")
            tests_passed += 1
        else:
            print(f"  ❌ {dir_name}/ - Directory not found")
            tests_failed += 1
    
    # Check __init__.py files
    init_files = [
        "data_scrapers/__init__.py",
        "features/__init__.py",
        "model/__init__.py",
    ]
    
    for init_file in init_files:
        if os.path.isfile(init_file):
            print(f"  ✅ {init_file}")
            tests_passed += 1
        else:
            print(f"  ❌ {init_file} - File not found")
            tests_failed += 1
    
    return tests_passed, tests_failed

def test_config():
    """Test config file"""
    print("\n🔍 Testing configuration...\n")
    
    try:
        import config
        
        required_vars = [
            'STOCK_NAME',
            'START_DATE',
            'END_DATE',
            'OUTPUT_DIR',
            'FEATURE_CSV_PATH',
            'PREDICTION_LOG_PATH',
        ]
        
        tests_passed = 0
        tests_failed = 0
        
        for var in required_vars:
            if hasattr(config, var):
                value = getattr(config, var)
                print(f"  ✅ {var} = {value}")
                tests_passed += 1
            else:
                print(f"  ❌ {var} - Not found in config.py")
                tests_failed += 1
        
        return tests_passed, tests_failed
        
    except ImportError:
        print("  ❌ config.py not found")
        return 0, 1

def test_data_fetch():
    """Quick test of data fetching"""
    print("\n🔍 Testing data fetching (quick test)...\n")
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Try to fetch 5 days of AAPL data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        print("  Fetching 5 days of AAPL data...")
        ticker = yf.Ticker("AAPL")
        df = ticker.history(start=start_date, end=end_date)
        
        if not df.empty:
            print(f"  ✅ Successfully fetched {len(df)} days of data")
            print(f"  ✅ Latest close: ${df['Close'].iloc[-1]:.2f}")
            return 2, 0
        else:
            print("  ❌ No data returned")
            return 0, 1
            
    except Exception as e:
        print(f"  ❌ Data fetch failed: {e}")
        return 0, 1

def test_sentiment():
    """Quick test of sentiment analysis"""
    print("\n🔍 Testing sentiment analysis...\n")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        test_texts = [
            ("Stock prices are soaring!", "positive"),
            ("Market crash imminent", "negative"),
            ("Trading sideways", "neutral"),
        ]
        
        tests_passed = 0
        
        for text, expected in test_texts:
            sentiment = analyzer.polarity_scores(text)
            score = sentiment['compound']
            
            if expected == "positive" and score > 0:
                result = "✅"
                tests_passed += 1
            elif expected == "negative" and score < 0:
                result = "✅"
                tests_passed += 1
            elif expected == "neutral" and abs(score) < 0.1:
                result = "✅"
                tests_passed += 1
            else:
                result = "❌"
            
            print(f"  {result} '{text}' → {score:.3f} ({expected})")
        
        return tests_passed, len(test_texts) - tests_passed
        
    except Exception as e:
        print(f"  ❌ Sentiment test failed: {e}")
        return 0, 1

def main():
    """Run all tests"""
    print("="*60)
    print("PROJECT KASSANDRA - INSTALLATION TEST")
    print("="*60)
    print()
    
    total_passed = 0
    total_failed = 0
    
    # Run tests
    passed, failed = test_imports()
    total_passed += passed
    total_failed += failed
    
    passed, failed = test_directories()
    total_passed += passed
    total_failed += failed
    
    passed, failed = test_config()
    total_passed += passed
    total_failed += failed
    
    passed, failed = test_data_fetch()
    total_passed += passed
    total_failed += failed
    
    passed, failed = test_sentiment()
    total_passed += passed
    total_failed += failed
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n  Total Passed: {total_passed}")
    print(f"  Total Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n  ✅ ALL TESTS PASSED!")
        print("  ✅ You're ready to run: python main.py")
        print()
        return 0
    else:
        print("\n  ⚠️  SOME TESTS FAILED")
        print("  ⚠️  Please fix the issues above before running main.py")
        print("\n  Common fixes:")
        print("    - Run: pip install -r requirements.txt")
        print("    - Create missing directories")
        print("    - Check internet connection")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())