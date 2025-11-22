import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class EquityCurve:
    def __init__(self, initial_equity: float, signal: pd.Series, log_returns: pd.Series):
        self.initial_equity: float = initial_equity
        self.signal: pd.Series = signal
        self.log_returns: pd.Series = log_returns
        self.equity_values: pd.Series = np.exp(
            log_returns.cumsum()) * initial_equity
        self.dates: pd.DatetimeIndex = log_returns.index

    def plot(self, label: str = "Equity Curve", ax: plt.Figure = None):
        # only equity changes when a trade occurs so check that the signal changes
        trade_mask = self.signal.ne(self.signal.shift())
        plotted_equity_values = self.equity_values[trade_mask]
        plotted_times = self.dates[trade_mask]

        if ax:
            plt.plot(plotted_times, plotted_equity_values, label=label)

        else:
            plt.figure(figsize=(12, 6))
            plt.plot(plotted_times, plotted_equity_values, label=label)
            plt.xlabel("Date")
            plt.ylabel("Equity Value")
            plt.title("Equity Curve")
            plt.legend()
            plt.grid(True)
            plt.show()

    @staticmethod
    def combine_equity_curves(equity_curves: list, labels: list):
        if len(equity_curves) != len(labels):
            raise ValueError(
                "Number of equity curves must match the number of labels.")

        figure = plt.figure(figsize=(12, 6))
        for curve, label in zip(equity_curves, labels):
            curve.plot(label, figure)

        plt.xlabel("Date")
        plt.ylabel("Equity Value")
        plt.title("Combined Equity Curves")
        plt.legend()
        plt.grid(True)
        plt.show()
