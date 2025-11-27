import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from Engine.Utils import dollar_format


class SkewCurve:

    def __init__(self, trade_curve: np.ndarray, trade_indicies: np.ndarray, intital_equity: float = 100_000):
        self.trade_curve: np.ndarray = trade_curve
        self.trade_indicies: np.ndarray = trade_indicies
        self.initial_equity: float = intital_equity

        # 0 0 0 0 1 0 0 0 1
        # 0 0 0 0 1 1 1 1 2
        trade_pnl_series = pd.Series(self.trade_curve)
        trade_groups: np.ndarray = self.trade_indicies.cumsum()
        self.trade_returns: pd.Series = trade_pnl_series.groupby(
            trade_groups).sum()
        self.trade_pct_returns: pd.Series = self.trade_returns / self.initial_equity * 100

    def plot(self, bins: int = 150):
        data = self.trade_pct_returns.dropna()

        mu, sigma = data.mean(), data.std()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_yscale('log')

        # Split positive & negative returns
        positive = data[data >= 0]
        negative = data[data < 0]

        # Common bin edges for both
        all_counts, bin_edges = np.histogram(data, bins=bins)

        # Plot negative (red)
        ax.hist(
            negative,
            bins=bin_edges,
            alpha=0.6,
            edgecolor='black',
            color='red',
            label="Negative Returns"
        )

        # Plot positive (green)
        ax.hist(
            positive,
            bins=bin_edges,
            alpha=0.6,
            edgecolor='black',
            color='green',
            label="Positive Returns"
        )

        # Vertical lines
        ax.axvline(0, color='black', linestyle='-',
                   linewidth=1.5, label="Zero")
        ax.axvline(mu, color='blue', linestyle='--',
                   linewidth=1.5, label=f"Mean = {mu:.5f}")

        ax.set_title(
            f"Trade Return Skew Distribution\nSkewness = {skew(data):.4f}")
        ax.set_xlabel(
            f"Percent Return per Trade\nInitial Equity: {dollar_format(self.initial_equity)}")
        ax.set_ylabel("Frequency")

        ax.legend()
        plt.show()
