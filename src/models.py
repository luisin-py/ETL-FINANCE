
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

class StockModel:
    def __init__(self, ticker, model_type, look_back=30):
        self.ticker = ticker
        self.model_type = model_type
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_path = os.path.join(MODELS_DIR, f"{ticker}_{model_type}.pkl")
        self.scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{model_type}_scaler.pkl")

    def prepare_data(self, data):
        """Prepares data for training."""
        data = data.sort_values('date')
        dataset = data['close'].values.reshape(-1, 1)
        
        # Fit scaler on all data for simplicity in this pipeline context, 
        # or properly split. Standard path: fit on train.
        training_size = int(len(dataset) * 0.70)
        train_data = dataset[0:training_size, :]
        
        self.scaler.fit(train_data)
        scaled_full = self.scaler.transform(dataset)
        
        return scaled_full, dataset

    def create_sequences(self, dataset):
        X, y = [], []
        if len(dataset) <= self.look_back:
            return np.array(X), np.array(y)
            
        for i in range(self.look_back, len(dataset)):
            X.append(dataset[i-self.look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        raise NotImplementedError

    def train(self, data, epochs=None):
        # Epochs arg is kept for compatibility but not used by all sklearn models
        scaled_full, raw_data = self.prepare_data(data)
        
        X, y = self.create_sequences(scaled_full)
        
        if len(X) == 0:
            print(f"Not enough data to train {self.ticker}")
            return
            
        # Split into train/test for validation printing
        train_size = int(len(X) * 0.85)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.model = self.build_model()
        
        self.model.fit(X_train, y_train)
        
        # Validation score
        if len(X_test) > 0:
            preds = self.model.predict(X_test)
            score = mean_squared_error(y_test, preds)
            print(f"{self.model_type} MSE on Test: {score:.5f}")
        
        self.save()

    def save(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            # print(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict_future(self, data, days=30):
        """Predicts the next N days."""
        if not self.model:
             if not self.load():
                return None
        
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # Start with the last look_back days
        curr_seq = scaled_data[-self.look_back:].reshape(1, -1)
        
        future_preds = []
        
        # Iterative prediction
        for _ in range(days):
            pred = self.model.predict(curr_seq)
            future_preds.append(pred[0])
            
            # Update sequence: shift left and append new prediction
            # curr_seq is [1, look_back]
            new_seq = np.append(curr_seq[0][1:], pred)
            curr_seq = new_seq.reshape(1, -1)
            
        future_preds = np.array(future_preds).reshape(-1, 1)
        return self.scaler.inverse_transform(future_preds)


class LSTMModel(StockModel):
    def __init__(self, ticker):
        # Fallback to GradientBoosting as LSTM (Tensorflow) is not stable on Python 3.13
        super().__init__(ticker, 'LSTM')

    def build_model(self):
        # Gradient Boosting is a strong time-series performer
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

class MLPModel(StockModel):
    def __init__(self, ticker):
        super().__init__(ticker, 'MLP')
        
    def build_model(self):
        # MLP Regressor from Scikit-Learn
        return MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
