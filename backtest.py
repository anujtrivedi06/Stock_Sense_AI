# backtester.py
"""
Backtesting engine for StockSenseAI
Simulates real trading using model signals day by day
"""

import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, initial_capital=10000, transaction_cost=0.001, threshold=0.0):
        """
        initial_capital : starting portfolio value in dollars
        transaction_cost: cost per trade as a fraction (0.001 = 0.1%)
        threshold       : minimum predicted return (absolute) to act on a signal
                          e.g. 0.005 = only trade if model predicts > +0.5% or < -0.5%
                          predictions inside the band → HOLD
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.threshold = threshold

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def run(self, combined_df, predictor, feature_columns, test_size=0.2):
        """
        Run a full backtest on the test portion of combined_df.

        Steps
        -----
        1. Split into train / test the same way the main pipeline does.
        2. For every day in the test set, ask the model: up or down tomorrow?
        3. Execute a simulated trade based on that signal.
        4. Track portfolio value day by day.
        5. Compute performance metrics and return everything.
        """

        # ---- 1. Split data (mirror main pipeline) ----
        split_idx = int(len(combined_df) * (1 - test_size))
        test_df = combined_df.iloc[split_idx:].copy().reset_index(drop=True)

        # ---- 2. Get model predictions (returns) for the test period ----
        X_test = test_df[feature_columns]
        predicted_returns = predictor.predict(X_test)

        # ---- 3. Simulate trading ----
        trade_log, portfolio_values = self._simulate_trading(
            test_df, predicted_returns
        )

        # ---- 4. Compute metrics ----
        metrics = self._compute_metrics(
            portfolio_values, test_df['close'].values, trade_log
        )

        # ---- 5. Build daily portfolio DataFrame for charting ----
        portfolio_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'Close': test_df['close'].values,
            'Predicted_Return': predicted_returns,
            'Portfolio_Value': portfolio_values,
            'Signal': [t['signal'] for t in trade_log]
        })

        return metrics, portfolio_df, trade_log

    # ------------------------------------------------------------------
    # TRADING SIMULATION
    # ------------------------------------------------------------------
    def _simulate_trading(self, test_df, predicted_returns):
        """
        Walk through every day in test_df, make a trade decision,
        and track the portfolio.

        Rules
        -----
        - If predicted return > +threshold  → BUY (go long if not already long)
        - If predicted return < -threshold  → SELL (exit position if holding)
        - If predicted return is inside the band → HOLD (not confident enough)
        - Only 1 position at a time (fully in or fully out)
        - Transaction cost applied on every buy and sell
        """

        cash = self.initial_capital
        shares_held = 0
        portfolio_values = []
        trade_log = []

        prices = test_df['close'].values
        dates = test_df['Date'].values

        for i in range(len(test_df)):
            price = prices[i]
            pred_return = predicted_returns[i]

            # ---- Apply confidence threshold ----
            # Only act if the predicted move is large enough to be meaningful
            if pred_return > self.threshold:
                signal = 'BUY'
            elif pred_return < -self.threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'  # Inside the noise band — do nothing

            # ---- Execute trade ----
            trade_executed = None

            if signal == 'BUY' and shares_held == 0 and cash > 0:
                # Buy as many shares as possible
                cost = price * (1 + self.transaction_cost)
                shares_bought = cash / cost
                shares_held = shares_bought
                cash = 0
                trade_executed = {
                    'date': dates[i],
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_held,
                    'predicted_return': pred_return,
                    'signal': signal
                }

            elif signal == 'SELL' and shares_held > 0:
                # Sell all shares
                proceeds = shares_held * price * (1 - self.transaction_cost)
                cash = proceeds
                trade_executed = {
                    'date': dates[i],
                    'action': 'SELL',
                    'price': price,
                    'shares': shares_held,
                    'predicted_return': pred_return,
                    'signal': signal
                }
                shares_held = 0

            else:
                # Hold current position — no trade
                trade_executed = {
                    'date': dates[i],
                    'action': 'HOLD',
                    'price': price,
                    'shares': shares_held,
                    'predicted_return': pred_return,
                    'signal': signal
                }

            trade_log.append(trade_executed)

            # ---- Portfolio value today ----
            portfolio_value = cash + (shares_held * price)
            portfolio_values.append(portfolio_value)

        return trade_log, portfolio_values

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _compute_metrics(self, portfolio_values, close_prices, trade_log):
        """
        Compute all performance metrics.
        """

        portfolio_values = np.array(portfolio_values)
        close_prices = np.array(close_prices)

        # ---- Returns ----
        strategy_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        buyhold_return = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100

        # ---- Daily returns ----
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # ---- Sharpe Ratio (annualised, risk-free rate ~0) ----
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # ---- Max Drawdown ----
        peak = portfolio_values[0]
        max_drawdown = 0
        for v in portfolio_values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # ---- Win Rate ----
        buy_trades = [t for t in trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in trade_log if t['action'] == 'SELL']

        wins = 0
        losses = 0
        profit_per_trade = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            pnl = (sell_price - buy_price) / buy_price * 100
            profit_per_trade.append(pnl)
            if pnl > 0:
                wins += 1
            else:
                losses += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        avg_win = np.mean([p for p in profit_per_trade if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in profit_per_trade if p < 0]) if losses > 0 else 0

        # ---- Profit Factor ----
        gross_profit = sum(p for p in profit_per_trade if p > 0)
        gross_loss = abs(sum(p for p in profit_per_trade if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        return {
            # Settings used
            'Threshold_Used': self.threshold,

            # Returns
            'Strategy_Return_%': round(strategy_return, 2),
            'BuyHold_Return_%': round(buyhold_return, 2),
            'Alpha_%': round(strategy_return - buyhold_return, 2),

            # Risk
            'Sharpe_Ratio': round(sharpe, 3),
            'Max_Drawdown_%': round(max_drawdown, 2),

            # Trading stats
            'Total_Trades': total_trades,
            'Win_Rate_%': round(win_rate, 2),
            'Avg_Win_%': round(avg_win, 2),
            'Avg_Loss_%': round(avg_loss, 2),
            'Profit_Factor': round(profit_factor, 3),

            # Capital
            'Initial_Capital': self.initial_capital,
            'Final_Capital': round(portfolio_values[-1], 2),
        }