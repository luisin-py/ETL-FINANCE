
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from db import init_db, save_data

# List of tickers to track (Major B3 stocks)
TICKERS = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA',
    'ABEV3.SA', 'WEGE3.SA', 'RENT3.SA', 'BPAC11.SA', 'EQTL3.SA'
]

def run_etl():
    """Runs the ETL process."""
    print("Starting ETL process...")
    
    # Initialize database
    init_db()
    
    # Define date range: Last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730) # approx 2 years
    
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Basic cleaning if necessary
                # yfinance returns MultiIndex columns sometimes if multiple tickers, 
                # but here we fetch one by one.
                
                # Verify columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in data.columns for col in required_cols):
                    save_data(data, ticker)
                else:
                    print(f"Warning: Missing columns for {ticker}. Columns found: {data.columns}")
            else:
                print(f"No data found for {ticker}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    print("ETL process completed.")

if __name__ == "__main__":
    run_etl()
