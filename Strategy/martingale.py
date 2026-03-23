# martingale.py
"""
Martingale Trading Strategy
After each loss, double position size. Reset on win.
"""
import pandas as pd
import numpy as np

class MartingaleBacktester:
    def __init__(
        self,
        initial_capital: float = 10_000,
        base_bet_pct: float = 0.05,      # 5% of capital as base bet
        max_multiplier: int = 8,          # Cap: base × 8 (3 doublings max)
        transaction_cost: float = 0.001,  # 0.1% per trade
        signal_threshold: float = 0.005,  # Min predicted move to trade
    ):
        self.initial_capital = initial_capital
        self.base_bet_pct = base_bet_pct
        self.max_multiplier = max_multiplier
        self.transaction_cost = transaction_cost
        self.signal_threshold = signal_threshold

    def run(self, combined_df, predictor, feature_columns, test_size=0.2):
        df = combined_df.copy().reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        capital = self.initial_capital
        multiplier = 1         # Current bet multiplier
        portfolio_values = []
        trades = []

        for i in range(len(test_df) - 1):
            row = test_df.iloc[i]
            next_row = test_df.iloc[i + 1]

            features = test_df[feature_columns].iloc[[i]]
            predicted_return = predictor.predict(features)[0]

            current_price = row["close"]
            next_price = next_row["close"]
            actual_return = (next_price - current_price) / current_price

            # --- Determine signal ---
            if predicted_return > self.signal_threshold:
                signal = "BUY"
            elif predicted_return < -self.signal_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"

            trade_return = 0.0
            if signal != "HOLD":
                # Bet size = base_pct × multiplier, capped
                bet_pct = self.base_bet_pct * min(multiplier, self.max_multiplier)
                bet_amount = capital * bet_pct

                if signal == "BUY":
                    trade_return = actual_return - self.transaction_cost
                else:  # SELL / short
                    trade_return = -actual_return - self.transaction_cost

                pnl = bet_amount * trade_return
                capital += pnl
                capital = max(capital, 0)  # Floor at zero

                won = pnl > 0

                # --- Martingale rule ---
                if won:
                    multiplier = 1               # Reset on win
                else:
                    multiplier = min(multiplier * 2, self.max_multiplier)

                trades.append({
                    "Date": row["Date"],
                    "Signal": signal,
                    "Multiplier": min(multiplier, self.max_multiplier),
                    "Bet_Pct": round(bet_pct * 100, 2),
                    "PnL": round(pnl, 2),
                    "Won": won,
                    "Capital_After": round(capital, 2),
                })

            portfolio_values.append(capital)

        portfolio_df = pd.DataFrame({
            "Date": test_df["Date"].iloc[:len(portfolio_values)],
            "Capital": portfolio_values,
        })

        trade_log = pd.DataFrame(trades)

        metrics = self._compute_metrics(portfolio_df, trade_log)
        return metrics, portfolio_df, trade_log

    def _compute_metrics(self, portfolio_df, trade_log):
        final = portfolio_df["Capital"].iloc[-1]
        strategy_return = (final - self.initial_capital) / self.initial_capital * 100

        wins = trade_log[trade_log["Won"] == True]
        losses = trade_log[trade_log["Won"] == False]
        win_rate = len(wins) / max(len(trade_log), 1) * 100

        total_won = wins["PnL"].sum() if not wins.empty else 0
        total_lost = abs(losses["PnL"].sum()) if not losses.empty else 1
        profit_factor = total_won / max(total_lost, 1)

        rolling_max = portfolio_df["Capital"].cummax()
        drawdown = (portfolio_df["Capital"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        daily_returns = portfolio_df["Capital"].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                  if daily_returns.std() > 0 else 0)

        max_streak = 0
        streak = 0
        for won in trade_log["Won"]:
            if not won:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return {
            "Initial_Capital": self.initial_capital,
            "Final_Capital": round(final, 2),
            "Strategy_Return_%": round(strategy_return, 2),
            "Win_Rate_%": round(win_rate, 1),
            "Profit_Factor": round(profit_factor, 3),
            "Max_Drawdown_%": round(max_drawdown, 2),
            "Sharpe_Ratio": round(sharpe, 3),
            "Total_Trades": len(trade_log),
            "Max_Loss_Streak": max_streak,
            "Max_Multiplier_Used": int(trade_log["Multiplier"].max()) if not trade_log.empty else 1,
        }