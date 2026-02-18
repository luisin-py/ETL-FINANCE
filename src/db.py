
import sqlite3
import pandas as pd
import os
import json

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'finance.db')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(cursor):
    """Creates all necessary tables."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(date, ticker)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            execution_date TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            model_type TEXT NOT NULL,
            predicted_price REAL,
            set_type TEXT,
            UNIQUE(ticker, prediction_date, model_type, set_type)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_optimization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_date TEXT NOT NULL,
            weights_json TEXT NOT NULL,
            return REAL,
            volatility REAL,
            sharpe_ratio REAL,
            model_source TEXT,
            UNIQUE(execution_date, model_source)
        )
    ''')

def init_db():
    """Initializes the database with the required schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    create_tables(cursor)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_data(df, ticker):
    """Saves the dataframe to the database."""
    if df.empty:
        return

    conn = get_db_connection()
    df = df.reset_index()
    
    if 'Date' in df.columns:
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    df['ticker'] = ticker
    
    # Check if required columns exist before selecting
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if not available_cols:
        print(f"No valid data columns found for {ticker}")
        conn.close()
        return

    # Select available columns plus date and ticker
    cols_to_select = ['date', 'ticker'] + available_cols
    data_to_save = df[cols_to_select].copy()
    
    # Rename columns to lowercase
    new_cols = ['date', 'ticker'] + [col.lower() for col in available_cols]
    data_to_save.columns = new_cols
    
    try:
        # Use a temporary table or manual deletion to handle upserts more cleanly
        cursor = conn.cursor()
        min_date = data_to_save['date'].min()
        max_date = data_to_save['date'].max()
        
        cursor.execute("DELETE FROM stock_prices WHERE ticker = ? AND date BETWEEN ? AND ?", (ticker, min_date, max_date))
        data_to_save.to_sql('stock_prices', conn, if_exists='append', index=False)
        print(f"Updated {len(df)} records for {ticker}.")
    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")
        
    conn.commit()
    conn.close()

def get_data(ticker=None, start_date=None, end_date=None):
    """Retrieves data from the database."""
    conn = get_db_connection()
    query = "SELECT * FROM stock_prices WHERE 1=1"
    params = []
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
        
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
        
    query += " ORDER BY date ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        
    return df

def get_available_tickers():
    """Returns a list of unique tickers in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        tickers = []
    conn.close()
    return tickers

def save_prediction(predictions_df):
    """Saves predictions to the database."""
    if predictions_df.empty:
        return
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Delete existing predictions for the same ticker, model, type and dates to avoid constraints issues
        # A simpler approach is to try insert, if fails (integrity), delete and insert.
        # But 'to_sql' with replace drops the whole table. 'append' fails on unique.
        
        # Iterating might be slow but safe for now, or just delete matching range.
        for _, row in predictions_df.iterrows():
            cursor.execute('''
                DELETE FROM predictions 
                WHERE ticker = ? AND prediction_date = ? AND model_type = ? AND set_type = ?
            ''', (row['ticker'], row['prediction_date'], row['model_type'], row['set_type']))
        
        predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
        print(f"Saved {len(predictions_df)} predictions.")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    conn.commit()
    conn.close()

def get_predictions(ticker, model_type=None):
    """Retrieves predictions for a ticker."""
    conn = get_db_connection()
    query = "SELECT * FROM predictions WHERE ticker = ?"
    params = [ticker]
    
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
        
    query += " ORDER BY prediction_date ASC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def save_portfolio_optimization(execution_date, weights, ret, vol, sharpe, model_source):
    """Saves portfolio optimization results."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    weights_json = json.dumps(weights)
    
    try:
        cursor.execute("DELETE FROM portfolio_optimization WHERE execution_date = ? AND model_source = ?", (execution_date, model_source))
        cursor.execute('''
            INSERT INTO portfolio_optimization (execution_date, weights_json, return, volatility, sharpe_ratio, model_source)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (execution_date, weights_json, ret, vol, sharpe, model_source))
        print(f"Saved portfolio optimization for {execution_date} ({model_source}).")
    except Exception as e:
        print(f"Error saving portfolio: {e}")
        
    conn.commit()
    conn.close()

def get_latest_portfolio(model_source=None):
    """Retrieves the latest portfolio optimization."""
    conn = get_db_connection()
    query = "SELECT * FROM portfolio_optimization"
    params = []
    
    if model_source:
        query += " WHERE model_source = ?"
        params.append(model_source)
        
    query += " ORDER BY execution_date DESC LIMIT 1"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df.iloc[0] if not df.empty else None
