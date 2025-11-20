import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Data.data_loader import DataTimeseries
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


class Chart:

    def __init__(self, data_timeseries: DataTimeseries, window: int = 300):
        self.data_timeseries = data_timeseries
        self.window = window
        self.plot_start = len(self.data_timeseries.df) - self.window

        self.fig, self.ax = plt.subplots(figsize=(16, 8))

    def plot_candles(self):
        # self.ax.clear()
        print(self.window)
        print(self.plot_start)
        print(self.data_timeseries.df.size)
        window_df = self.data_timeseries.df[self.plot_start:self.plot_start + self.window]
        print(window_df, window_df.size)
        print(window_df.index)
        self.ax.plot(window_df.index,
                     window_df['Close'], color='cyan', linewidth=1.5)
        # self.ax.set_title("Close Price")
        # self.ax.set_xlabel("Time")
        # self.ax.set_ylabel("Price")
        # self.ax.xaxis.set_major_formatter(
        #     mdates.DateFormatter('%Y-%m-%d %H:%M'))
        # self.fig.autofmt_xdate()
        # self.ax.set_facecolor('#1e1e1e')  # Dark theme
        # self.ax.tick_params(colors='white')
        # self.ax.yaxis.label.set_color('white')
        # self.ax.xaxis.label.set_color('white')
        # self.ax.title.set_color('white')
        # self.ax.plot([1, 2, 3], [4, 5, 6])

    def show(self):
        plt.show()
