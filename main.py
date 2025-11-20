from Data.data_loader import DataTimeseries
from Graphics.chart import Chart

if __name__ == '__main__':
    nq_timeseries_data = DataTimeseries('./Data/data_combined.csv')
    chart = Chart(nq_timeseries_data)
    chart.plot_candles()
    chart.show()
