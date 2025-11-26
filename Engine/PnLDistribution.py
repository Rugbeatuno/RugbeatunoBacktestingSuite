import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from Engine.Utils import dollar_format


class SkewCurve:

    def __init__(self, trade_curve: pd.Series, trade_indicies: pd.Series, intital_equity: float = 100_000):
        self.trade_curve: pd.Series = trade_curve
        self.trade_indicies: pd.Series = trade_indicies
        self.initial_equity: float = intital_equity

        # 0 0 0 0 1 0 0 0 1
        # 0 0 0 0 1 1 1 1 2
        self.trade_groups: pd.Series = self.trade_indicies.cumsum()
        self.trade_returns = self.trade_curve.groupby(
            self.trade_groups).cumsum()
        self.trade_pct_returns = self.trade_returns / intital_equity * 100

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

        # Normal curve overlay
        # x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        # pdf = norm.pdf(x, mu, sigma)

        # # Convert PDF → histogram units
        # N = len(data)
        # bin_width = bin_edges[1] - bin_edges[0]
        # scaled_pdf = pdf * N * bin_width

        # ax.plot(
        #     x, scaled_pdf,
        #     linestyle='dotted',
        #     linewidth=2,
        #     color='blue',
        #     label="Normal PDF (scaled)"
        # )

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
