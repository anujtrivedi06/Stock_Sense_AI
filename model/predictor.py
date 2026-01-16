# model/predictor.py
"""
Machine Learning predictor using ensemble approach
Now includes train error diagnostics
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
            'rf': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=5, min_samples_split=10),
            'gb': GradientBoostingRegressor(n_estimators=50, random_state=42, learning_rate=0.05, max_depth=3),
            'xgb': XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1, learning_rate=0.05, max_depth=3)
        }
        self.weights = {'rf': 0.3, 'gb': 0.3, 'xgb': 0.4}
        self.is_trained = False

        # Store training data for diagnostics
        self.X_train_scaled = None
        self.y_train = None

    def train(self, X_train, y_train):
        """
        Train ensemble of models and store training data for diagnostics
        """
        print("\n🤖 Training models...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Save for later train-metric evaluation
        self.X_train = X_train
        self.y_train = y_train


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

    def _compute_metrics(self, y_true, y_pred):
        """
        Helper function to compute metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Direction accuracy
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100

        return rmse, mae, mape, direction_accuracy

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on BOTH train and test sets
        """
        # ----- TEST METRICS -----
        test_predictions = self.predict(X_test)

        test_rmse, test_mae, test_mape, test_dir = self._compute_metrics(
            y_test, test_predictions
        )

        # ----- TRAIN METRICS -----
        train_predictions = self.predict(self.X_train)
        

        train_rmse, train_mae, train_mape, train_dir = self._compute_metrics(
            self.y_train, train_predictions
        )

        print("\n📊 MODEL EVALUATION SUMMARY")

        print("\n--- TRAIN METRICS ---")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Train MAPE: {train_mape:.4f}")
        print(f"  Train Direction Accuracy: {train_dir:.4f}%")

        print("\n--- TEST METRICS ---")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test MAPE: {test_mape:.4f}")
        print(f"  Test Direction Accuracy: {test_dir:.4f}%")

        # ----- OVERFITTING GAP -----
        print("\n--- OVERFITTING DIAGNOSTICS ---")
        print(f"  RMSE Gap (Test - Train): {test_rmse - train_rmse:.4f}")
        print(f"  MAE Gap (Test - Train): {test_mae - train_mae:.4f}")
        print(f"  Direction Gap: {test_dir - train_dir:.4f}%")

        metrics = {
            'Train_RMSE': train_rmse,
            'Train_MAE': train_mae,
            'Train_MAPE': train_mape,
            'Train_Direction_Accuracy': train_dir,

            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_MAPE': test_mape,
            'Test_Direction_Accuracy': test_dir,

            'RMSE_Gap': test_rmse - train_rmse,
            'MAE_Gap': test_mae - train_mae
        }

        return metrics, test_predictions

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
