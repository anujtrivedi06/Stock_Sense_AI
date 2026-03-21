# model/general_predictor.py
"""
GeneralPredictor — a StockPredictor trained on multiple stocks simultaneously.

Identical interface to StockPredictor so it's a drop-in replacement.
The only difference is it knows it was trained on a universe of stocks
and carries metadata about that universe for display purposes.

Usage:
    gp = GeneralPredictor()
    gp.train(X_train, y_train, universe=['AAPL', 'TSLA', ...])
    gp.save_model('outputs/models/general_model.pkl')

    # Later, for any stock — no retraining needed:
    gp = GeneralPredictor()
    gp.load_model('outputs/models/general_model.pkl')
    prediction = gp.predict(X_features)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from datetime import datetime


class GeneralPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=0.6,
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                random_state=42,
                learning_rate=0.02,
                max_depth=2,
                subsample=0.8,
                min_samples_leaf=10,
            ),
            'xgb': XGBRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                learning_rate=0.02,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=2.0,
                verbosity=0,
            ),
        }
        self.weights     = {'rf': 0.3, 'gb': 0.3, 'xgb': 0.4}
        self.is_trained  = False
        self.universe    = []        # stocks this model was trained on
        self.trained_at  = None      # timestamp
        self.n_samples   = 0         # total training rows
        self.feature_columns = []    # stored so app can verify alignment

        # stored for overfitting diagnostics
        self.X_train = None
        self.y_train = None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, universe=None):
        """
        Train on combined multi-stock data.
        universe: list of ticker strings used to build the training set.
        """
        print(f"\n🌍 Training General Model...")
        if universe:
            print(f"   Universe: {', '.join(universe)}")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]}")

        # Hard fill — GradientBoosting and XGBoost reject NaN outright
        if X_train.isna().any().any():
            X_train = X_train.fillna(0)
        if hasattr(y_train, 'isna') and y_train.isna().any():
            raise ValueError("Target y_train contains NaN — check feature engineering.")

        X_scaled = self.scaler.fit_transform(X_train)
        self.X_train = X_train
        self.y_train = y_train

        for name, model in self.models.items():
            print(f"   Training {name.upper()}...")
            model.fit(X_scaled, y_train)

        self.is_trained      = True
        self.universe        = universe or []
        self.trained_at      = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.n_samples       = len(X_train)
        self.feature_columns = list(X_train.columns)

        print(f"✓ General model trained on {len(self.universe)} stocks "
              f"({self.n_samples:,} samples)")

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("General model not trained yet!")
        X_scaled = self.scaler.transform(X)
        preds = {name: model.predict(X_scaled)
                 for name, model in self.models.items()}
        return (self.weights['rf']  * preds['rf'] +
                self.weights['gb']  * preds['gb'] +
                self.weights['xgb'] * preds['xgb'])

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #
    def evaluate(self, X_test, y_test):
        test_preds  = self.predict(X_test)
        train_preds = self.predict(self.X_train)

        def metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            ad   = np.diff(y_true.values if hasattr(y_true, 'values') else y_true) > 0
            pd_  = np.diff(y_pred) > 0
            da   = np.mean(ad == pd_) * 100
            return rmse, mae, da

        tr_rmse, tr_mae, tr_da = metrics(self.y_train, train_preds)
        te_rmse, te_mae, te_da = metrics(y_test,       test_preds)

        print("\n📊 GENERAL MODEL EVALUATION")
        print(f"  Train  — RMSE: {tr_rmse:.4f}  MAE: {tr_mae:.4f}  Dir: {tr_da:.1f}%")
        print(f"  Test   — RMSE: {te_rmse:.4f}  MAE: {te_mae:.4f}  Dir: {te_da:.1f}%")
        print(f"  Gap    — RMSE: {te_rmse-tr_rmse:+.4f}  Dir: {te_da-tr_da:+.1f}%")

        return {
            'Train_RMSE': tr_rmse, 'Train_MAE': tr_mae, 'Train_DA': tr_da,
            'Test_RMSE':  te_rmse, 'Test_MAE':  te_mae, 'Test_DA':  te_da,
            'RMSE_Gap':   te_rmse - tr_rmse,
        }, test_preds

    # ------------------------------------------------------------------ #
    #  Feature importance                                                  #
    # ------------------------------------------------------------------ #
    def get_feature_importance(self, top_n=20):
        """Returns top N features by Random Forest importance."""
        if not self.is_trained or not self.feature_columns:
            return pd.Series(dtype=float)
        imp = self.models['rf'].feature_importances_
        return (pd.Series(imp, index=self.feature_columns)
                  .sort_values(ascending=False)
                  .head(top_n))

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            'models':          self.models,
            'scaler':          self.scaler,
            'weights':         self.weights,
            'is_trained':      self.is_trained,
            'universe':        self.universe,
            'trained_at':      self.trained_at,
            'n_samples':       self.n_samples,
            'feature_columns': self.feature_columns,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"✓ General model saved → {filepath}  ({size_mb:.1f} MB)")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        self.models          = payload['models']
        self.scaler          = payload['scaler']
        self.weights         = payload['weights']
        self.is_trained      = payload['is_trained']
        self.universe        = payload.get('universe', [])
        self.trained_at      = payload.get('trained_at', 'unknown')
        self.n_samples       = payload.get('n_samples', 0)
        self.feature_columns = payload.get('feature_columns', [])
        print(f"✓ General model loaded  ({self.n_samples:,} training samples, "
              f"{len(self.universe)} stocks, trained {self.trained_at})")

    def is_available(self, filepath):
        """Quick check — does a saved general model exist at filepath?"""
        return os.path.exists(filepath)