
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

from db import get_data, save_prediction, save_portfolio_optimization, get_available_tickers
from models import LSTMModel, MLPModel, MODELS_DIR
from portfolio import optimize_portfolio

def calculate_expected_return(current_price, future_price, days):
    """Calculates annualized expected return."""
    if current_price == 0:
        return 0
    return ((future_price - current_price) / current_price) * (252 / days)

def run_training_pipeline():
    print("Starting Training Pipeline...")
    
    tickers = get_available_tickers()
    if not tickers:
        print("No tickers found in database. Run ETL first.")
        return

    # Dictionary to store latest historical data for portfolio optimization
    historical_data_dict = {}
    
    # Dictionaries to store expected returns from each model type for portfolio optimization
    lstm_returns = {}
    mlp_returns = {}
    
    execution_date = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in tickers:
        print(f"\n--- Processing {ticker} ---")
        try:
            df = get_data(ticker)
            if df.empty or len(df) < 100:
                print(f"Not enough data for {ticker}")
                continue
                
            historical_data_dict[ticker] = df
            current_price = df['close'].iloc[-1]
            last_date = df['date'].iloc[-1]
            
            # --- Train and Predict LSTM ---
            print(f"Training LSTM for {ticker}...")
            lstm = LSTMModel(ticker)
            lstm.train(df, epochs=20) # Reduced epochs for faster execution in this env
            
            # Generate predictions on Test set (last 15% of data)
            # To get dates for test set predictions, we need to map indices back to DF
            # The model.predict returns predictions matching the input sequences
            # Input sequences start from look_back index
            
            # Predict Future (Next 30 days)
            future_days = 30
            future_preds_lstm = lstm.predict_future(df, days=future_days)
            
            if future_preds_lstm is not None:
                # Save Future Predictions
                future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
                
                pred_df_lstm = pd.DataFrame({
                    'ticker': ticker,
                    'execution_date': execution_date,
                    'prediction_date': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'model_type': 'LSTM',
                    'predicted_price': future_preds_lstm.flatten(),
                    'set_type': 'Forecast'
                })
                save_prediction(pred_df_lstm)
                
                # Calculate Expected Return (using 21 days ~ 1 month for stability)
                target_idx = min(21, len(future_preds_lstm)-1)
                price_forecast = future_preds_lstm[target_idx][0]
                exp_ret = calculate_expected_return(current_price, price_forecast, target_idx+1)
                lstm_returns[ticker] = exp_ret
                print(f"LSTM Expected Return: {exp_ret:.4f}")

            # --- Train and Predict MLP ---
            print(f"Training MLP for {ticker}...")
            mlp = MLPModel(ticker)
            mlp.train(df, epochs=20)
            
            # Predict Future
            future_preds_mlp = mlp.predict_future(df, days=future_days)
            
            if future_preds_mlp is not None:
                pred_df_mlp = pd.DataFrame({
                    'ticker': ticker,
                    'execution_date': execution_date,
                    'prediction_date': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'model_type': 'MLP',
                    'predicted_price': future_preds_mlp.flatten(),
                    'set_type': 'Forecast'
                })
                save_prediction(pred_df_mlp)
                
                target_idx = min(21, len(future_preds_mlp)-1)
                price_forecast = future_preds_mlp[target_idx][0]
                exp_ret = calculate_expected_return(current_price, price_forecast, target_idx+1)
                mlp_returns[ticker] = exp_ret
                print(f"MLP Expected Return: {exp_ret:.4f}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

    # --- Run Portfolio Optimization ---
    print("\n--- Running Portfolio Optimization ---")
    
    # 1. Markowitz with Historical Mean (Benchmark)
    # print("Optimizing with Historical Returns...")
    # res_hist = optimize_portfolio(tickers, historical_data_dict, predicted_returns_dict=None)
    # if res_hist:
    #     save_portfolio_optimization(execution_date, res_hist['weights'], res_hist['return'], 
    #                                res_hist['volatility'], res_hist['sharpe_ratio'], 'Historical')

    # 2. Markowitz with LSTM Predictions
    if lstm_returns:
        print("Optimizing with LSTM Returns...")
        res_lstm = optimize_portfolio(list(lstm_returns.keys()), historical_data_dict, predicted_returns_dict=lstm_returns)
        if res_lstm:
            save_portfolio_optimization(execution_date, res_lstm['weights'], res_lstm['return'], 
                                       res_lstm['volatility'], res_lstm['sharpe_ratio'], 'LSTM')
            print("Saved LSTM Portfolio.")

    # 3. Markowitz with MLP Predictions
    if mlp_returns:
        print("Optimizing with MLP Returns...")
        res_mlp = optimize_portfolio(list(mlp_returns.keys()), historical_data_dict, predicted_returns_dict=mlp_returns)
        if res_mlp:
            save_portfolio_optimization(execution_date, res_mlp['weights'], res_mlp['return'], 
                                       res_mlp['volatility'], res_mlp['sharpe_ratio'], 'MLP')
            print("Saved MLP Portfolio.")

    print("\nTraining Pipeline Completed.")

if __name__ == "__main__":
    run_training_pipeline()
