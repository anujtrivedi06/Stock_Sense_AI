# backtester.py
"""
Backtesting engine for StockSenseAI — with proper StopLossManager integration.

Logic:
  - BUY  : predicted return > +threshold AND not in position
  - SELL : predicted return < -threshold AND in position  (ML exit)
  - STOP : price hits stop level → forced exit regardless of ML signal
  - Winners ride until ML says SELL or stop trails up and triggers
  - Last day → always close any open position
  - No dumb timer — stop loss does the risk management properly
"""

import pandas as pd
import numpy as np
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from stop_loss import StopLossManager, StopLossConfig, compute_atr
from .stop_loss import StopLossManager, StopLossConfig, compute_atr


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10_000,
        transaction_cost: float = 0.001,
        threshold: float = 0.005,
        sl_config: StopLossConfig = None,
    ):
        """
        initial_capital  : starting portfolio value in dollars
        transaction_cost : fraction per trade (0.001 = 0.1%)
        threshold        : minimum |predicted return| to act on a signal
        sl_config        : StopLossConfig instance — if None, defaults to
                           fixed 3% stop loss with a 5% hard floor
        """
        self.initial_capital  = initial_capital
        self.transaction_cost = transaction_cost
        self.threshold        = threshold
        self.sl_config        = sl_config or StopLossConfig(
            mode="fixed",
            fixed_pct=0.03,
            hard_floor_pct=0.05,
        )

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def run(self, combined_df, predictor, feature_columns, test_size=0.2):

        split_idx = int(len(combined_df) * (1 - test_size))
        test_df   = combined_df.iloc[split_idx:].copy().reset_index(drop=True)

        predicted_returns = predictor.predict(test_df[feature_columns])

        # Pre-compute ATR series for the test window (needed for ATR stop mode)
        atr_series = compute_atr(
            test_df['high'], test_df['low'], test_df['close'],
            window=self.sl_config.atr_window
        )

        trade_log, portfolio_values = self._simulate_trading(
            test_df, predicted_returns, atr_series
        )

        metrics = self._compute_metrics(
            portfolio_values, test_df['close'].values, trade_log
        )

        portfolio_df = pd.DataFrame({
            'Date':             test_df['Date'].values,
            'Close':            test_df['close'].values,
            'Predicted_Return': predicted_returns,
            'Portfolio_Value':  portfolio_values,
            'Signal':           [t['signal']  for t in trade_log],
            'Action':           [t['action']  for t in trade_log],
            'Stop_Price':       [t.get('stop_price') for t in trade_log],
        })

        return metrics, portfolio_df, trade_log

    # ------------------------------------------------------------------
    # TRADING SIMULATION
    # ------------------------------------------------------------------
    def _simulate_trading(self, test_df, predicted_returns, atr_series):
        """
        Day-by-day simulation with StopLossManager handling all exits.

        Exit priority (highest to lowest):
          1. Stop loss triggered  → STOP_EXIT   (loss cut immediately)
          2. ML SELL signal       → SELL         (model says exit)
          3. Last day             → FINAL_EXIT   (clean close)
          4. Otherwise            → HOLD
        """
        cash        = self.initial_capital
        shares_held = 0.0
        sl          = StopLossManager(self.sl_config)

        portfolio_values = []
        trade_log        = []

        prices = test_df['close'].values
        dates  = test_df['Date'].values
        n      = len(test_df)

        for i in range(n):
            price       = prices[i]
            pred_return = predicted_returns[i]
            is_last_day = (i == n - 1)
            atr_val     = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else None

            # ---- ML signal ----
            if pred_return > self.threshold:
                signal = 'BUY'
            elif pred_return < -self.threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            action     = 'HOLD'
            stop_price = None

            # ---- If in position: check stop first, then ML exit ----
            if shares_held > 0:
                # Update trailing stop upward if price rose
                sl.update_trailing(price, atr_val)

                # Check if stop is triggered
                stop_triggered, stop_px, stop_reason = sl.check(price)
                stop_price = stop_px if stop_px else sl.stop_price

                if stop_triggered:
                    # Stop loss hit — exit immediately, ignore ML signal
                    proceeds    = shares_held * price * (1 - self.transaction_cost)
                    cash        = proceeds
                    shares_held = 0.0
                    action      = 'STOP_EXIT'
                    sl.reset()

                elif signal == 'SELL' or is_last_day:
                    # ML says sell, or it's the last day — clean exit
                    proceeds    = shares_held * price * (1 - self.transaction_cost)
                    cash        = proceeds
                    shares_held = 0.0
                    action      = 'SELL' if signal == 'SELL' else 'FINAL_EXIT'
                    sl.reset()

                else:
                    action = 'HOLD'

            # ---- If not in position: check for BUY signal ----
            elif signal == 'BUY' and cash > 0 and not is_last_day:
                cost        = price * (1 + self.transaction_cost)
                shares_held = cash / cost
                cash        = 0.0
                action      = 'BUY'
                sl.open_position(price, entry_atr=atr_val)
                stop_price  = sl.stop_price

            trade_log.append({
                'date':             dates[i],
                'action':           action,
                'signal':           signal,
                'price':            price,
                'shares':           shares_held,
                'predicted_return': pred_return,
                'stop_price':       stop_price,
            })

            portfolio_values.append(cash + shares_held * price)

        return trade_log, portfolio_values

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _compute_metrics(self, portfolio_values, close_prices, trade_log):

        portfolio_values = np.array(portfolio_values)
        close_prices     = np.array(close_prices)

        strategy_return = (
            (portfolio_values[-1] - self.initial_capital)
            / self.initial_capital * 100
        )
        buyhold_return = (
            (close_prices[-1] - close_prices[0])
            / close_prices[0] * 100
        )

        # Sharpe — only on days where portfolio actually moved
        daily_returns  = np.diff(portfolio_values) / portfolio_values[:-1]
        active_returns = daily_returns[np.abs(daily_returns) > 1e-8]

        if len(active_returns) > 1 and active_returns.std() > 0:
            sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown — from initial capital, not first day value
        peak         = self.initial_capital
        max_drawdown = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

        # Trade P&L — pair each BUY with the next exit of any kind
        buy_trades  = [t for t in trade_log if t['action'] == 'BUY']
        exit_trades = [t for t in trade_log
                       if t['action'] in ('SELL', 'STOP_EXIT', 'FINAL_EXIT')]

        profit_per_trade = []
        exit_types       = []
        for i in range(min(len(buy_trades), len(exit_trades))):
            buy_price  = buy_trades[i]['price']
            exit_price = exit_trades[i]['price']
            pnl        = (exit_price - buy_price) / buy_price * 100
            profit_per_trade.append(pnl)
            exit_types.append(exit_trades[i]['action'])

        wins   = sum(1 for p in profit_per_trade if p > 0)
        losses = sum(1 for p in profit_per_trade if p <= 0)
        total_trades = wins + losses

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_win  = np.mean([p for p in profit_per_trade if p > 0])  if wins   > 0 else 0.0
        avg_loss = np.mean([p for p in profit_per_trade if p <= 0]) if losses > 0 else 0.0

        gross_profit  = sum(p for p in profit_per_trade if p > 0)
        gross_loss    = abs(sum(p for p in profit_per_trade if p <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        stop_exits  = exit_types.count('STOP_EXIT')
        ml_exits    = exit_types.count('SELL')
        final_exits = exit_types.count('FINAL_EXIT')

        return {
            'Threshold_Used':    self.threshold,
            'Stop_Mode':         self.sl_config.mode.upper(),

            'Strategy_Return_%': round(strategy_return, 2),
            'BuyHold_Return_%':  round(buyhold_return, 2),
            'Alpha_%':           round(strategy_return - buyhold_return, 2),

            'Sharpe_Ratio':      round(sharpe, 3),
            'Max_Drawdown_%':    round(max_drawdown, 2),

            'Total_Trades':      total_trades,
            'Stop_Exits':        stop_exits,
            'ML_Exits':          ml_exits,
            'Final_Exits':       final_exits,
            'Win_Rate_%':        round(win_rate, 2),
            'Avg_Win_%':         round(avg_win, 2),
            'Avg_Loss_%':        round(avg_loss, 2),
            'Profit_Factor':     round(profit_factor, 3),

            'Initial_Capital':   self.initial_capital,
            'Final_Capital':     round(portfolio_values[-1], 2),
        }