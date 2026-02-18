
import numpy as np
import pandas as pd
import json

def optimize_portfolio(tickers, historical_data_dict, predicted_returns_dict=None, num_portfolios=10000, risk_free_rate=0.0):
    """
    Optimizes portfolio using Markowitz theory and Monte Carlo simulation.
    
    Args:
        tickers: List of ticker symbols.
        historical_data_dict: Dict mapping ticker to historical DataFrame (must have 'close').
        predicted_returns_dict: Dict mapping ticker to expected return (float). 
                               If None, uses historical mean return.
    """
    
    # alignment
    valid_tickers = [t for t in tickers if t in historical_data_dict and not historical_data_dict[t].empty]
    
    if len(valid_tickers) < 2:
        print("Not enough valid tickers for optimization.")
        return None
        
    # Create combined close dataframe for covariance
    df_close = pd.DataFrame()
    for t in valid_tickers:
        df_close[t] = historical_data_dict[t].set_index('date')['close']
        
    df_close = df_close.dropna()
    
    if df_close.empty:
        print("No overlapping data for covariance calculation.")
        return None
        
    # Calculate Log Returns for Covariance
    log_returns = np.log(df_close / df_close.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252
    
    # Expected Returns
    # If predicted_returns_dict is provided, use those. Else use historical.
    expected_returns = []
    
    if predicted_returns_dict:
        # User requested to use ML predictions for expected return
        # predicted_return should be an annual return ideally, or period return.
        # Assuming predicted_returns_dict provides the PROJECTED RETURN for the next period (e.g. 1 month or 1 year)
        # We need to standardize this.
        # Let's assume the passed value is the raw expected return (e.g. 0.05 for 5%)
        for t in valid_tickers:
            expected_returns.append(predicted_returns_dict.get(t, 0.0))
    else:
        # Use historical mean
        expected_returns = log_returns.mean() * 252
        expected_returns = expected_returns.values
    
    expected_returns = np.array(expected_returns)
    
    # Monte Carlo Simulation
    num_assets = len(valid_tickers)
    all_weights = np.zeros((num_portfolios, num_assets))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for i in range(num_portfolios):
        weights = np.array(np.random.random(num_assets))
        weights /= np.sum(weights)
        all_weights[i,:] = weights
        
        # Portfolio Return
        ret_arr[i] = np.sum(expected_returns * weights) 
        # Note: If expected_returns are annualized, this is annual. 
        
        # Portfolio Volatility
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe Ratio
        if vol_arr[i] == 0:
            sharpe_arr[i] = 0
        else:
            sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]

    # Find stats
    max_sharpe_idx = np.argmax(sharpe_arr)
    min_vol_idx = np.argmin(vol_arr)
    
    max_sharpe_weights = all_weights[max_sharpe_idx,:]
    max_sharpe_ret = ret_arr[max_sharpe_idx]
    max_sharpe_vol = vol_arr[max_sharpe_idx]
    max_sharpe_ratio = sharpe_arr[max_sharpe_idx]
    
    # Construct result dictionary
    weights_dict = {valid_tickers[i]: float(max_sharpe_weights[i]) for i in range(num_assets)}
    
    return {
        "weights": weights_dict,
        "return": float(max_sharpe_ret),
        "volatility": float(max_sharpe_vol),
        "sharpe_ratio": float(max_sharpe_ratio),
        "all_portfolios": {
            "returns": ret_arr.tolist(),
            "volatility": vol_arr.tolist(),
            "sharpe_ratio": sharpe_arr.tolist()
        }
    }
