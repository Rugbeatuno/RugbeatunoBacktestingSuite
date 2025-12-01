import pandas_ta as ta
from Engine.DataTimeseries import DataTimeseries
from Engine.Backtester import Backtester
from Graphics.Chart import Chart
from Engine.EquityCurve import EquityCurve
from Engine.Utils import plot_price_series

import time

import pandas as pd
import numpy as np

EMSD_SHORT_WEIGHT = 0.7
EMSD_LONG_WEIGHT = 0.3
EMSD_SHORT_LENGTH = 64
EMSD_LONG_LENGTH = 256

data = DataTimeseries('./Data/nq_continous.csv', 'NQ')
data.convert_timezone('America/New_York')
data.resample_data('5min')
close: pd.Series = data.get_close_prices_series()

# # N and 4N pairs
ema_8 = ta.ema(close, 8)
ema_32 = ta.ema(close, 32)

ema_16 = ta.ema(close, 16)
ema_64 = ta.ema(close, 64)

ema_256 = ta.ema(close, 256)
# # print(close.ewm(span=EMSD_SHORT_LENGTH).std())
# # emsd = close.ewm(span=EMSD_SHORT_LENGTH).std() * EMSD_SHORT_WEIGHT + close.ewm(span=EMSD_LONG_LENGTH).std() * EMSD_LONG_WEIGHT

# # drop all nan values and sync them up
sync_df = pd.DataFrame({
    'ema8': ema_8,
    'ema16': ema_16,
    'ema32': ema_32,
    'ema64': ema_64,
    'ema256': ema_256,
}).fillna(0)


ema_8 = sync_df['ema8']
ema_16 = sync_df['ema16']
ema_32 = sync_df['ema32']
ema_64 = sync_df['ema64']
ema_256 = sync_df['ema256']


# # calculate forecasts
signal_8_32 = (ema_8 - ema_32)
signal_16_64 = (ema_16 - ema_64)
signal_64_256 = (ema_64 - ema_256)

volitility_8_32 = signal_8_32.ewm(span=EMSD_SHORT_LENGTH).std(
) * EMSD_SHORT_WEIGHT + signal_8_32.ewm(span=EMSD_LONG_LENGTH).std() * EMSD_LONG_WEIGHT
volitility_16_64 = signal_16_64.ewm(span=EMSD_SHORT_LENGTH).std(
) * EMSD_SHORT_WEIGHT + signal_16_64.ewm(span=EMSD_LONG_LENGTH).std() * EMSD_LONG_WEIGHT
volitility_64_256 = signal_64_256.ewm(span=EMSD_SHORT_LENGTH).std(
) * EMSD_SHORT_WEIGHT + signal_64_256.ewm(span=EMSD_LONG_LENGTH).std() * EMSD_LONG_WEIGHT

raw_forecast_8_32 = signal_8_32 / volitility_8_32
raw_forecast_16_64 = signal_16_64 / volitility_16_64
raw_forecast_64_256 = signal_64_256 / volitility_64_256


df = pd.DataFrame({
    '8_32': raw_forecast_8_32,
    '16_64': raw_forecast_16_64,
    '64_256': raw_forecast_64_256,
    '8_ema': ema_8,
    '16_ema': ema_16,
    '32_ema': ema_32,
    '64_ema': ema_64,
    '256_ema': ema_256,
    'close': close
})

np_8_32 = df['8_32'].to_numpy(dtype=np.float32)
np_16_64 = df['16_64'].to_numpy(dtype=np.float32)
np_64_256 = df['64_256'].to_numpy(dtype=np.float32)

clip_min = -20
clip_max = 20
np_8_32 = np.clip(np_8_32, clip_min, clip_max)
np_16_64 = np.clip(np_16_64, clip_min, clip_max)
np_64_256 = np.clip(np_64_256, clip_min, clip_max)

# trials = 2000

w1 = -18.15451124782517
w2 = -8.637803526827886
w3 = -3.527284884065461
w4 = 0.02169602541391937

combined_forecast = np_8_32 * w1 + np_16_64 * w2 + np_64_256 * w3
combined_forecast = combined_forecast / clip_max
signal = np.zeros_like(combined_forecast, dtype=np.float32)
signal[combined_forecast > w4] = -1
signal[combined_forecast < -w4] = 1
signal_series = pd.Series(signal, index=df.index)

# signal_series = pd.Series(np.ones(len(signal)))
print(len(signal_series))
backtester = Backtester(
    data.get_ohlc_prices(),
    data.indicies,
    signal_series.values,
    timeframe='1min',
    tick_value=0.5,
    contract_multiplier=2.0,
    initial_equity=100_000,
    slippage_ticks=0,
    commission=0

)

backtester.full_backtest()

print(backtester)
backtester.generate_trade_time_distribution()

# permuted_ohlc = data.get_permutation()

# plot_price_series([pd.Series(permuted_ohlc[:, 3], index=data.get_indicies()), data.get_close_prices_series()], [
#                   'permuted close', 'original'], pct_of_bars=1)
# permuted_series = DataTimeseries.generate_permuted_close_series(
#     data.get_close_prices(), data.get_indicies())
# plot_price_series([permuted_series, close], [
#                   'permuted', 'std'], pct_of_bars=0.2)

# plot_price_series([close], ['close'], pct_of_bars=.2)

# strategy_ret = backtester.generate_equity_curve()
# bh_ret = backtester.generate_buy_hold_equity_curve()

# strategy_ret.plot('strategy', show_drawdown=True)
# backtester.generate_skew_curve().plot()
# bh_ret.plot('buy hold', show_drawdown=True)

# EquityCurve.combine_equity_curves(
#     [strategy_ret, bh_ret], ["Strategy", "Buy & Hold"])

# backtester.generate_skew_curve().plot()
backtester.monte_carlo_pnl_resampling_test(iterations=1000, pct_of_trades=0.01)
backtester.monte_carlo_permutation_test(100, 0.01, 'Walk Forward')

# chart = Chart(data.get_ohlc_prices(), data.get_indicies(),
#               signal=signal_series.values, window=300)
# chart.plot_candles()
