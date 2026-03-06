# stop_loss.py
"""
Stop Loss Manager for StockSenseAI
Supports three modes:
  1. Fixed %     – exit if price drops X% from entry
  2. Trailing    – stop tracks peak price, exits on X% pullback
  3. ATR-based   – stop is set at N × ATR below entry
Stop loss ALWAYS overrides the ML signal.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StopLossConfig:
    mode: Literal["fixed", "trailing", "atr"] = "fixed"

    # Fixed / Trailing
    fixed_pct:    float = 0.02   # 2 % below entry
    trailing_pct: float = 0.02   # 2 % below peak

    # ATR
    atr_multiplier: float = 2.0  # stop = entry - N × ATR
    atr_window:     int   = 14   # ATR look-back window

    # Hard floor applied on top of all modes
    hard_floor_pct: Optional[float] = None   # e.g. 0.05 → never lose > 5 %


# ─────────────────────────────────────────────────────────────────────────────
# ATR helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Core stateful manager  (use one instance per open trade)
# ─────────────────────────────────────────────────────────────────────────────
class StopLossManager:
    """
    Example usage:
        sl = StopLossManager(config)
        sl.open_position(entry_price, entry_atr=atr_value)

        for each new bar:
            sl.update_trailing(current_price, current_atr)
            triggered, stop_px, reason = sl.check(current_price)
            if triggered:
                # close position – stop overrides everything
    """

    def __init__(self, config: StopLossConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self.entry_price: Optional[float] = None
        self.peak_price:  Optional[float] = None
        self.stop_price:  Optional[float] = None
        self.is_open: bool = False

    # ── Open a new long position ──────────────────────────────────────────
    def open_position(self, entry_price: float,
                      entry_atr: Optional[float] = None):
        self.entry_price = entry_price
        self.peak_price  = entry_price
        self.is_open     = True
        self.stop_price  = self._compute_stop(entry_price, entry_atr)

    # ── Update trailing stop upward each bar ─────────────────────────────
    def update_trailing(self, current_price: float,
                        current_atr: Optional[float] = None):
        if not self.is_open:
            return

        if self.cfg.mode == "trailing":
            if current_price > self.peak_price:
                self.peak_price = current_price
                self.stop_price = self._compute_stop(self.peak_price, current_atr)

        elif self.cfg.mode == "atr" and current_atr:
            if current_price > self.peak_price:
                self.peak_price = current_price
                new_stop = self.peak_price - self.cfg.atr_multiplier * current_atr
                if new_stop > self.stop_price:   # never lower the stop
                    self.stop_price = new_stop

    # ── Check if stop is triggered ───────────────────────────────────────
    def check(self, current_price: float) -> tuple:
        """Returns (triggered: bool, stop_price: float, reason: str)."""
        if not self.is_open or self.stop_price is None:
            return False, 0.0, ""

        if current_price <= self.stop_price:
            self.is_open = False
            pct = (current_price - self.entry_price) / self.entry_price * 100
            reason = (
                f"🛑 STOP LOSS HIT | Mode: {self.cfg.mode.upper()} | "
                f"Entry: ${self.entry_price:.2f} → Current: ${current_price:.2f} | "
                f"Stop level: ${self.stop_price:.2f} | Loss: {pct:.2f}%"
            )
            return True, self.stop_price, reason

        return False, self.stop_price, ""

    # ── Snapshot dict for Streamlit display ──────────────────────────────
    def snapshot(self) -> dict:
        if not self.is_open or self.entry_price is None:
            return {}
        return {
            "mode":           self.cfg.mode.upper(),
            "entry_price":    self.entry_price,
            "stop_price":     self.stop_price,
            "peak_price":     self.peak_price,
            "pct_from_entry": (self.stop_price - self.entry_price) / self.entry_price * 100,
        }

    # ── Private ───────────────────────────────────────────────────────────
    def _compute_stop(self, ref: float, atr: Optional[float]) -> float:
        cfg = self.cfg
        if cfg.mode == "fixed":
            stop = ref * (1 - cfg.fixed_pct)
        elif cfg.mode == "trailing":
            stop = ref * (1 - cfg.trailing_pct)
        elif cfg.mode == "atr":
            stop = ref - cfg.atr_multiplier * atr if atr else ref * (1 - cfg.fixed_pct)
        else:
            stop = ref * (1 - cfg.fixed_pct)

        if cfg.hard_floor_pct and self.entry_price:
            stop = max(stop, self.entry_price * (1 - cfg.hard_floor_pct))
        return stop


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised version — used by the backtester
# ─────────────────────────────────────────────────────────────────────────────
def apply_stop_loss_to_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    config: StopLossConfig
) -> pd.DataFrame:
    """
    Adjusts a raw signal series (+1 BUY / -1 SELL) with stop-loss logic.
    Returns df copy with extra columns:
        final_signal, stop_price, stop_triggered, stop_reason
    """
    atr_series = compute_atr(df["high"], df["low"], df["close"], config.atr_window)

    final_signals  = signals.copy().astype(float)
    stop_prices    = pd.Series(np.nan,  index=df.index)
    stop_triggered = pd.Series(False,   index=df.index)
    stop_reasons   = pd.Series("",      index=df.index)

    sl = StopLossManager(config)
    in_position = False

    for i in range(len(df)):
        price   = df["close"].iloc[i]
        atr_v   = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else None
        raw_sig = signals.iloc[i]

        if in_position:
            sl.update_trailing(price, atr_v)
            triggered, sp, reason = sl.check(price)

            if triggered:
                final_signals.iloc[i]  = -1
                stop_triggered.iloc[i] = True
                stop_reasons.iloc[i]   = reason
                stop_prices.iloc[i]    = sp
                in_position = False
                sl.reset()
                continue

            stop_prices.iloc[i] = sl.stop_price

            if raw_sig == -1:           # ML says sell too → exit
                in_position = False
                sl.reset()

        else:
            if raw_sig == 1:
                sl.open_position(price, entry_atr=atr_v)
                in_position = True
                stop_prices.iloc[i] = sl.stop_price

    result = df.copy()
    result["final_signal"]   = final_signals
    result["stop_price"]     = stop_prices
    result["stop_triggered"] = stop_triggered
    result["stop_reason"]    = stop_reasons
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────
def make_stop_loss_config(
    mode: str = "fixed",
    fixed_pct: float = 0.02,
    trailing_pct: float = 0.02,
    atr_multiplier: float = 2.0,
    atr_window: int = 14,
    hard_floor_pct: Optional[float] = None
) -> StopLossConfig:
    return StopLossConfig(
        mode=mode,
        fixed_pct=fixed_pct,
        trailing_pct=trailing_pct,
        atr_multiplier=atr_multiplier,
        atr_window=atr_window,
        hard_floor_pct=hard_floor_pct
    )