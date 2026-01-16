# model/predictor.py
"""
Machine Learning predictor using ensemble approach
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os

class StockPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.weights = {'rf': 0.3, 'gb': 0.3, 'xgb': 0.4}
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train ensemble of models
        """
        print("\n🤖 Training models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train each model
        for name, model in self.models.items():
            print(f"  Training {name.upper()}...")
            model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        print("✓ All models trained successfully")
    
    def predict(self, X):
        """
        Make ensemble prediction
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Ensemble: weighted average
        ensemble_pred = (
            self.weights['rf'] * predictions['rf'] +
            self.weights['gb'] * predictions['gb'] +
            self.weights['xgb'] * predictions['xgb']
        )
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Direction accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
        
        print("\n📊 Model Evaluation:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics, predictions
    
    def save_model(self, filepath):
        """
        Save trained model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'weights': self.weights,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.weights = model_data['weights']
        self.is_trained = model_data['is_trained']
        
        print(f"✓ Model loaded from {filepath}")