import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# !todo add drawdown, max drawdown


class EquityCurve:
    def __init__(self, equity_curve: pd.Series, drawdown_curve: pd.Series, trade_indicies: pd.Series, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        if len(equity_curve) != len(trade_indicies):
            raise ValueError(
                "Length of equity_curve != length of trade_indicies.")

        self.start_date = start_date if start_date else equity_curve.index[0]
        self.end_date = end_date if end_date else equity_curve.index[-1]

        self.trade_indicies: pd.Series = trade_indicies.loc[self.start_date: self.end_date]
        self.equity_values: pd.Series = equity_curve.loc[self.start_date: self.end_date]
        self.drawdown_curve: pd.Series = drawdown_curve.loc[self.start_date: self.end_date]
        self.dates: pd.DatetimeIndex = self.equity_values.index

    def plot(self, label: str = "Equity Curve", ax: plt.Figure = None, show_drawdown: bool = False):
        # Only plot points where trades occur
        mask = self.trade_indicies.astype(bool)
        plotted_equity_values = self.equity_values[mask]
        plotted_times = self.dates[mask]

        if ax is None and not show_drawdown:
            # Normal one-plot mode
            plt.figure(figsize=(12, 6))
            plt.plot(plotted_times, plotted_equity_values, label=label)
            plt.xlabel("Date")
            plt.ylabel("Equity Value")
            plt.title(f"Equity Curve {self.start_date} to {self.end_date}")
            plt.legend()
            plt.grid(True)
            plt.show()
            return

        if not show_drawdown:
            # Provided axis, no drawdown subplot
            ax.plot(plotted_times, plotted_equity_values, label=label)
            return

        # ---- DRAW EQUITY + DRAWDOWN SUBPLOTS ----
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1]
        )

        # --- Top: Equity Curve ---
        ax1.plot(plotted_times, plotted_equity_values,
                 label=label, color="blue")
        ax1.set_ylabel("Equity Value")
        ax1.set_title(f"Equity Curve {self.start_date} to {self.end_date}")
        ax1.grid(True)
        ax1.legend()

        # --- Bottom: Drawdown Curve ---
        # Drawdown was passed in dollars → convert to percent
        drawdown_pct = self.drawdown_curve / self.equity_values.max() * 100
        drawdown_pct = drawdown_pct[mask]

        ax2.fill_between(plotted_times, drawdown_pct,
                         0, color="red", alpha=0.4)
        ax2.plot(plotted_times, drawdown_pct, color="red", linewidth=1.2)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def combine_equity_curves(equity_curves: list, labels: list):
        if len(equity_curves) != len(labels):
            raise ValueError(
                "Number of equity curves must match the number of labels.")

        figure, ax = plt.subplots(figsize=(12, 6))
        for curve, label in zip(equity_curves, labels):
            curve.plot(label, ax, show_drawdown=False)

        plt.xlabel("Date")
        plt.ylabel("Equity Value")
        plt.title(
            f"Combined Equity Curves {equity_curves[0].start_date} to {equity_curves[0].end_date}")
        plt.legend()
        plt.grid(True)
        plt.show()
