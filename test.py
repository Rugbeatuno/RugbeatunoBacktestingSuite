from Engine.DataTimeseries import DataTimeseries

data = DataTimeseries('./Data/nq_continous.csv', 'NQ')
print(data)
print(data.get_close_prices())
print(data.get_start_date())
print(data.get_end_date())
print(data.get_timeframe())
print(data.get_indicies())
print(data.get_ohlc_prices())
print(data.get_close_prices())
print(data.resample_data('1D'))
print(data)
