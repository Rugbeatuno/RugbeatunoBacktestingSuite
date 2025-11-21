import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from Data.DataTimeseries import DataTimeseries
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


class Chart:

    MAX_WINDOW = 1000
    MIN_WINDOW = 50

    def __init__(self, data_timeseries: DataTimeseries, window: int = 300):
        self.data_timeseries = data_timeseries
        self.window = window
        self.total_candles = len(self.data_timeseries.df)
        self.plot_start = max(0, self.total_candles - self.window)

        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.fig.canvas.mp

    def plot_candles(self):
        self.redraw()
        plt.show()

    def redraw(self):
        self.ax.clear()
        window_df = self.data_timeseries.df.iloc[self.plot_start:self.plot_start + self.window]
        mpf.plot(window_df, type='candle', ax=self.ax, volume=False)
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
