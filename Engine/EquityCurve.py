import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# !todo add drawdown, max drawdown


class EquityCurve:
    def __init__(self, equity_curve: pd.Series, trade_indicies: pd.Series, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        if len(equity_curve) != len(trade_indicies):
            raise ValueError(
                "Length of equity_curve != length of trade_indicies.")

        self.start_date = start_date if start_date else equity_curve.index[0]
        self.end_date = end_date if end_date else equity_curve.index[-1]

        self.trade_indicies: pd.Series = trade_indicies.loc[self.start_date: self.end_date]
        self.equity_values: pd.Series = equity_curve.loc[self.start_date: self.end_date]
        self.dates: pd.DatetimeIndex = self.equity_values.index

    def plot(self, label: str = "Equity Curve", ax: plt.Figure = None):
        # only equity changes when a trade occurs so check that the signal changes
        plotted_equity_values = self.equity_values[self.trade_indicies.astype(
            bool)]
        plotted_times = self.dates[self.trade_indicies.astype(bool)]

        if ax:
            plt.plot(plotted_times, plotted_equity_values, label=label)

        else:
            plt.figure(figsize=(12, 6))
            plt.plot(plotted_times, plotted_equity_values, label=label)
            plt.xlabel("Date")
            plt.ylabel("Equity Value")
            plt.title(f"Equity Curve ({self.start_date} -> {self.end_date})")
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
        plt.title(
            f"Combined Equity Curves ({equity_curves[0].start_date} -> {equity_curves[0].end_date})")
        plt.legend()
        plt.grid(True)
        plt.show()
