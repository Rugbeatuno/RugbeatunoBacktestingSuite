import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from Engine.DataTimeseries import DataTimeseries


class Chart:

    MAX_WINDOW = 1000
    MIN_WINDOW = 50

    def __init__(self, ohlc: np.ndarray, indicies: pd.DatetimeIndex, signal: np.ndarray = None, trade_indicies: np.ndarray = None, window: int = 300):
        self.ohlc: np.ndarray = ohlc
        self.indicies: pd.DatetimeIndex = indicies
        self.signal: np.ndarray = signal
        self.trade_indicies: np.ndarray = trade_indicies
        self.window: int = window
        self.total_candles: int = len(self.ohlc)
        self.plot_start: int = max(0, self.total_candles - self.window)

        self.df = pd.DataFrame(self.ohlc, index=self.indicies, columns=[
            'Open', 'High', 'Low', 'Close'])

        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(4, 1, top=0.95, bottom=0.05)
        self.ax1 = self.fig.add_subplot(gs[:-1, 0])  # Top 3/4 for price
        self.ax2 = self.fig.add_subplot(
            gs[-1, 0])  # Bottom 1/4 for signal

        # Hide the X-axis labels from the price plot for cleaner look
        # plt.setp(self.ax1.get_xticklabels(), visible=False)

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.fig.canvas.mp

    def plot_candles(self):
        self.redraw()
        plt.show()

    def redraw(self):
        self.ax1.clear()
        self.ax2.clear()

        end_index = self.plot_start + self.window
        window_df = self.df.iloc[self.plot_start:end_index]
        window_indicies = self.indicies[self.plot_start:end_index]
        mpf.plot(
            window_df,
            type='candle',
            ax=self.ax1,
            volume=False,
            datetime_format='%Y-%m-%d %H:%M',
            xrotation=0,
            style='yahoo'  # Use a simple style
        )

        # # plot signal bars, bottom 5%
        # if self.signal is not None:
        #     window_signal = self.signal[self.plot_start:end_index]

        #     # Create color mapping: 1=green, -1=red, 0=gray/white
        #     colors = np.where(window_signal > 0, 'g', np.where(
        #         window_signal < 0, 'r', 'lightgray'))

        #     self.ax2.bar(
        #         x=window_indicies,
        #         height=window_signal,
        #         width=1/1440,  # width of one minute in days
        #         color=colors
        #     )

        #     self.ax2.set_yticks([-1, 0, 1])
        #     self.ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
        #     self.ax2.tick_params(axis='y', labelsize=8)
        #     self.ax2.set_xlabel("Date")
        #     self.ax2.grid(axis='y', linestyle='--')

        # # --- 4. Plot Trade Arrows (ax1) ---

        # if self.trade_indicies is not None:
        #     window_trades = self.trade_indicies[self.plot_start:end_index]

        #     # Filter for entry/exit points (trade_indicies != 0)
        #     trade_mask = (window_trades != 0)

        #     trade_dates = window_indicies[trade_mask]
        #     trade_prices = window_df['Close'][trade_mask]
        #     trade_values = window_trades[trade_mask]

        #     # Long entries (signal change to +1 or trade_indicies > 0)
        #     long_mask = (trade_values > 0)
        #     self.ax1.plot(
        #         trade_dates[long_mask],
        #         trade_prices[long_mask],
        #         '^', color='green', markersize=8, label='Long Entry', zorder=10
        #     )

        #     # Short entries (signal change to -1 or trade_indicies < 0)
        #     short_mask = (trade_values < 0)
        #     self.ax1.plot(
        #         trade_dates[short_mask],
        #         trade_prices[short_mask],
        #         'v', color='red', markersize=8, label='Short Entry', zorder=10
        #     )

        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            self.scroll_left()
        else:
            self.scroll_right()

        self.redraw()

    def scroll_left(self):
        if self.plot_start <= 0:
            return

        self.plot_start -= 5

    def scroll_right(self):
        if self.plot_start < self.total_candles - self.window:
            return

        self.plot_start += 5
