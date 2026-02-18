
import yfinance as yf
import json
import os

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sectors.json')

def get_stock_sector(ticker):
    """
    Retrieves the sector for a given ticker.
    Uses a local JSON cache to avoid repeated API calls.
    """
    # Load cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                cache = json.load(f)
            except:
                cache = {}
    else:
        cache = {}
        
    # Check cache
    if ticker in cache:
        return cache[ticker]
    
    # Fetch from API
    try:
        # Append .SA for Brazilian stocks if not present (assuming B3 context from user prompt)
        # But system uses tickers like 'PETR4.SA' likely.
        # Let's assume ticker format is correct.
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'Outros')
        
        # Update cache
        cache[ticker] = sector
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
            
        return sector
    except Exception as e:
        print(f"Error fetching sector for {ticker}: {e}")
        return "Desconhecido"
