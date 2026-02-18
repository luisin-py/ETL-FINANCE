# ETL Finance Dashboard

This project is a Python-based ETL (Extract, Transform, Load) pipeline and dashboard for analyzing Brazilian stock market data (B3). It fetches data for the last 2 years using `yfinance`, stores it in a local SQLite database, and visualizes it with a Streamlit dashboard.

## Features

- **ETL Pipeline**: Fetches data for major B3 stocks (e.g., PETR4, VALE3, ITUB4).
- **Local Database**: Stores data in SQLite for fast retrieval and persistence.
- **Interactive Dashboard**: Visualizes stock prices (Candlestick charts) and volume using Streamlit and Plotly.
- **Date Filtering**: Allows users to select custom date ranges.

## Prerequisites

- Python 3.8+
- pip

## Installation

1.  Clone the repository or download the source code.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Or install the package in editable mode:

    ```bash
    pip install -e .
    ```

## Usage

### 1. Run the ETL Pipeline

Before running the dashboard, you need to populate the database with stock data. Run the following command:

```bash
python src/etl.py
```

This will:
- Connect to `yfinance`.
- Download the last 2 years of data for the configured tickers.
- Save the data to `data/finance.db`.

### 2. Run the Dashboard

After the ETL process is complete, start the Streamlit dashboard:

```bash
streamlit run src/app.py
```

This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

## Project Structure

- `data/`: Contains the SQLite database (`finance.db`).
- `src/`: Source code.
    - `db.py`: Database connection and operations.
    - `etl.py`: Data extraction and loading logic.
    - `app.py`: Streamlit dashboard application.
- `requirements.txt`: Python dependencies.
- `setup.py`: Package installation script.

## Customization

To add more stocks, edit the `TICKERS` list in `src/etl.py`.
