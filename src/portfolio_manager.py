
import pandas as pd
import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
TRANSACTIONS_FILE = os.path.join(DATA_DIR, 'transactions.json')

class PortfolioManager:
    def __init__(self):
        self.transactions = self.load_transactions()

    def load_transactions(self):
        if os.path.exists(TRANSACTIONS_FILE):
            with open(TRANSACTIONS_FILE, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return []
        return []

    def save_transactions(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TRANSACTIONS_FILE, 'w') as f:
            json.dump(self.transactions, f, indent=4)

    def add_transaction(self, date, ticker, type_trans, quantity, price):
        """
        type_trans: 'COMPRA' or 'VENDA'
        """
        transaction = {
            "date": date.strftime("%Y-%m-%d"),
            "ticker": ticker.upper(),
            "type": type_trans,
            "quantity": float(quantity),
            "price": float(price)
        }
        self.transactions.append(transaction)
        self.save_transactions()

    def get_holdings(self):
        """
        Returns a dictionary or DataFrame of current holdings.
        {ticker: {'quantity': q, 'avg_price': p}}
        """
        holdings = {}
        
        # Sort transactions by date
        sorted_trans = sorted(self.transactions, key=lambda x: x['date'])
        
        for t in sorted_trans:
            ticker = t['ticker']
            qty = t['quantity']
            price = t['price']
            
            if ticker not in holdings:
                holdings[ticker] = {'quantity': 0.0, 'total_cost': 0.0, 'avg_price': 0.0}
            
            if t['type'] == 'COMPRA':
                holdings[ticker]['quantity'] += qty
                holdings[ticker]['total_cost'] += qty * price
            elif t['type'] == 'VENDA':
                # FIFO or Average Cost? Using Average Cost for simplicity
                # Creating a realized gain/loss logic would be complex.
                # Just reducing quantity and proportional cost.
                avg = holdings[ticker]['avg_price']
                holdings[ticker]['quantity'] -= qty
                holdings[ticker]['total_cost'] -= qty * avg # Reduce cost basis proportionally
            
            # Recalculate Avg Price
            if holdings[ticker]['quantity'] > 0:
                holdings[ticker]['avg_price'] = holdings[ticker]['total_cost'] / holdings[ticker]['quantity']
            else:
                holdings[ticker]['quantity'] = 0
                holdings[ticker]['total_cost'] = 0
                holdings[ticker]['avg_price'] = 0

        # Remove zero holdings
        holdings = {k: v for k, v in holdings.items() if v['quantity'] > 0.001}
        return holdings

    def get_suggestions(self, optimal_weights, current_prices):
        """
        optimal_weights: dict {ticker: weight} (0.0 to 1.0)
        current_prices: dict {ticker: price}
        """
        holdings = self.get_holdings()
        
        # Calculate Total Portfolio Value
        total_value = 0.0
        current_allocations = {}
        
        for ticker, data in holdings.items():
            price = current_prices.get(ticker, data['avg_price']) # Fallback to avg price
            val = data['quantity'] * price
            total_value += val
            current_allocations[ticker] = val

        # Handle Cash? Assuming fully invested for suggestion logic or just rebalance existing capital.
        # If total_value is 0 (new portfolio), we can't suggest based on % rebalance without cash input.
        # Let's assume we rebalance the CURRENT total value.
        
        if total_value == 0:
            return []

        suggestions = []
        
        # Union of all tickers (current + optimal)
        all_tickers = set(holdings.keys()) | set(optimal_weights.keys())
        
        for ticker in all_tickers:
            curr_weight = current_allocations.get(ticker, 0.0) / total_value
            target_weight = optimal_weights.get(ticker, 0.0)
            
            curr_price = current_prices.get(ticker, 0.0)
            if curr_price == 0:
                continue # Cannot trade without price
                
            diff_weight = target_weight - curr_weight
            diff_value = diff_weight * total_value
            
            action = "MANTER"
            qty_action = 0
            
            if diff_value > curr_price: # Buy threshold (at least 1 share approx)
                action = "COMPRAR"
                qty_action = diff_value / curr_price
            elif diff_value < -curr_price:
                action = "VENDER"
                qty_action = abs(diff_value / curr_price)
                
            suggestions.append({
                "ticker": ticker,
                "current_weight": curr_weight,
                "target_weight": target_weight,
                "action": action,
                "quantity": qty_action,
                "financial_diff": diff_value
            })
            
        return pd.DataFrame(suggestions)
