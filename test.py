import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta


from Engine.DataTimeseries import DataTimeseries
from Engine.Backtester import Backtester
from Engine.Utils import timeframe_to_minutes
from Graphics.Chart import Chart


data = DataTimeseries('./Data/nq_continous.csv')
data.convert_timezone('America/New_York')
data.resample_data('5min')
data.slice(start_date=pd.Timestamp(year=2018, month=1, day=1))
in_sample, out_of_sample = data.split(pd.Timestamp(year=2023, month=1, day=1))

df: pd.DataFrame = in_sample.get_ohlc_prices_series()

ema8 = ta.ema(df['Close'], length=8)
ema128 = ta.ema(df['Close'], length=128)
signal = np.where(ema8 > ema128, 1, -1)
# print(pd.Series(signal).value_counts())

signal = np.ones_like(signal)
bt = Backtester(
    in_sample.get_ohlc_prices(),
    in_sample.get_indicies(),
    signal,
    in_sample.get_timeframe(),
    tick_value=0.5,
    contract_multiplier=2,
    initial_equity=100_000,
    slippage_ticks=2,
    commission=0.25
)
bt.full_backtest()
print(bt)
