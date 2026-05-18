import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TickData:
    """
    Loader for raw tick data files.

    Expected format (semicolon + space delimited):
        YYYYMMDD HHMMSS nanoseconds;price;low;high;volume

    Produces OHLC bars via tick, volume, or dollar aggregation.
    """

    AGGREGATION_METHODS = ('tick', 'volume', 'dollar', 'time')

    def __init__(self, filepath: str, timezone: str = 'America/Chicago'):
        self._filepath = filepath
        self._timezone = timezone

        self._timestamps: np.ndarray = None  # datetime64[ns]
        self._prices: np.ndarray = None      # float64 last/trade price
        self._lows: np.ndarray = None        # float64 tick low
        self._highs: np.ndarray = None       # float64 tick high
        self._volumes: np.ndarray = None     # int64 tick volume

        # aggregated output
        self._ohlc: np.ndarray = None
        self._indices: pd.DatetimeIndex = None
        self._bar_method: str = None
        self._bar_size: int = None

        self._load_raw(filepath)

    def _load_raw(self, filepath: str):
        print(f'\nLoading tick data: {filepath}')

        df = pd.read_csv(
            filepath,
            sep=';',
            header=None,
            names=['ts_raw', 'price', 'low', 'high', 'volume'],
            engine='pyarrow',
            dtype_backend='pyarrow'
        )

        # Do NOT parse datetimes here. Just store the raw strings.
        self._ts_raw = df['ts_raw'].to_numpy()  # Store as an array of strings
        self._prices = df['price'].to_numpy()
        self._lows = df['low'].to_numpy()
        self._highs = df['high'].to_numpy()
        self._volumes = df['volume'].to_numpy()

        print(f'Loaded {len(self._prices):,} ticks')
        print(
            f'Price range: {self._prices.min():.2f} - {self._prices.max():.2f}\n')

    def aggregate(self, method: str = 'tick', size: int = 250) -> 'TickData':
        if method not in self.AGGREGATION_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Use: {self.AGGREGATION_METHODS}")

        self._bar_method = method
        self._bar_size = size

        if method == 'tick':
            self._aggregate_tick_bars(size)
        elif method == 'volume':
            self._aggregate_volume_bars(size)
        elif method == 'dollar':
            self._aggregate_dollar_bars(size)
        elif method == 'time':
            self._aggregate_time_bars(size)

        print(
            f'Aggregated {len(self._prices):,} ticks into {len(self._ohlc):,} bars ({size} {method})')
        return self

    def _aggregate_tick_bars(self, n_ticks: int):
        n_total = len(self._prices)
        n_bars = n_total // n_ticks

        trim = n_bars * n_ticks
        prices = self._prices[:trim].reshape(n_bars, n_ticks)
        highs = self._highs[:trim].reshape(n_bars, n_ticks)
        lows = self._lows[:trim].reshape(n_bars, n_ticks)

        # Reshape the raw string timestamps
        ts_raw = self._ts_raw[:trim].reshape(n_bars, n_ticks)

        opens = prices[:, 0]
        closes = prices[:, -1]
        bar_highs = highs.max(axis=1)
        bar_lows = lows.min(axis=1)

        # Get ONLY the closing string for each bar (reduces parsing by 250x)
        bar_close_strings = ts_raw[:, -1]

        self._ohlc = np.column_stack([opens, bar_highs, bar_lows, closes])

        # NOW parse the datetimes using standard Pandas string methods
        # This will take milliseconds because it's only 92,000 strings instead of 23,000,000
        ts_series = pd.Series(bar_close_strings)
        date_str = ts_series.str.slice(0, 8)
        time_str = ts_series.str.slice(9, 15)
        nanos_str = ts_series.str.slice(16)

        dt_str = date_str.str.slice(0, 4) + '-' + date_str.str.slice(4, 6) + '-' + date_str.str.slice(6, 8) + ' ' + \
            time_str.str.slice(0, 2) + ':' + time_str.str.slice(2,
                                                                4) + ':' + time_str.str.slice(4, 6)

        base_timestamps = pd.to_datetime(
            dt_str, format='%Y-%m-%d %H:%M:%S').values.astype('datetime64[ns]')
        nanos = pd.to_numeric(nanos_str, errors='coerce').fillna(
            0).astype(np.int64).values

        final_timestamps = base_timestamps + nanos.astype('timedelta64[ns]')
        self._indices = pd.DatetimeIndex(
            final_timestamps).tz_localize(self._timezone)

    def _aggregate_volume_bars(self, target_volume: int):
        cum_vol = np.cumsum(self._volumes)
        bar_ids = cum_vol // target_volume

        # find last tick index of each bar
        bar_boundaries = np.where(np.diff(bar_ids))[0]
        if len(bar_boundaries) == 0:
            raise ValueError(
                f"Not enough volume for even 1 bar at size={target_volume}")

        starts = np.concatenate([[0], bar_boundaries + 1])
        ends = np.concatenate([bar_boundaries, [len(self._prices) - 1]])
        n_bars = len(starts)

        opens = self._prices[starts]
        closes = self._prices[ends]
        bar_times = self._timestamps[ends]

        # for highs/lows we need per-bar max/min
        bar_highs = np.empty(n_bars, dtype=np.float64)
        bar_lows = np.empty(n_bars, dtype=np.float64)
        for i in range(n_bars):
            s, e = starts[i], ends[i] + 1
            bar_highs[i] = self._highs[s:e].max()
            bar_lows[i] = self._lows[s:e].min()

        self._ohlc = np.column_stack([opens, bar_highs, bar_lows, closes])
        self._indices = pd.DatetimeIndex(bar_times).tz_localize(self._timezone)

    def _aggregate_dollar_bars(self, target_dollars: float):
        dollar_values = self._prices * self._volumes.astype(np.float64)
        cum_dollars = np.cumsum(dollar_values)
        bar_ids = (cum_dollars // target_dollars).astype(np.int64)

        bar_boundaries = np.where(np.diff(bar_ids))[0]
        if len(bar_boundaries) == 0:
            raise ValueError(
                f"Not enough dollar volume for even 1 bar at size={target_dollars}")

        starts = np.concatenate([[0], bar_boundaries + 1])
        ends = np.concatenate([bar_boundaries, [len(self._prices) - 1]])
        n_bars = len(starts)

        opens = self._prices[starts]
        closes = self._prices[ends]
        bar_times = self._timestamps[ends]

        bar_highs = np.empty(n_bars, dtype=np.float64)
        bar_lows = np.empty(n_bars, dtype=np.float64)
        for i in range(n_bars):
            s, e = starts[i], ends[i] + 1
            bar_highs[i] = self._highs[s:e].max()
            bar_lows[i] = self._lows[s:e].min()

        self._ohlc = np.column_stack([opens, bar_highs, bar_lows, closes])
        self._indices = pd.DatetimeIndex(bar_times).tz_localize(self._timezone)

    def _aggregate_time_bars(self, seconds: int):
        # aggregate ticks into fixed time windows
        idx = pd.DatetimeIndex(self._timestamps).tz_localize(self._timezone)
        df = pd.DataFrame({
            'price': self._prices,
            'low': self._lows,
            'high': self._highs,
            'volume': self._volumes,
        }, index=idx)

        rule = f'{seconds}s'
        resampled = df.resample(rule, label='right', closed='left').agg({
            'price': ['first', 'last'],
            'high': 'max',
            'low': 'min',
        }).dropna()

        resampled.columns = ['Open', 'Close', 'High', 'Low']

        self._ohlc = resampled[['Open', 'High', 'Low', 'Close']].values
        self._indices = resampled.index

    def get_ohlc(self) -> np.ndarray:
        if self._ohlc is None:
            raise RuntimeError("Call .aggregate() before accessing OHLC data")
        return self._ohlc

    def get_indices(self) -> pd.DatetimeIndex:
        if self._indices is None:
            raise RuntimeError("Call .aggregate() before accessing indices")
        return self._indices

    def get_close_prices(self) -> np.ndarray:
        return self.get_ohlc()[:, 3]

    def get_close_prices_series(self) -> pd.Series:
        return pd.Series(self.get_close_prices(), index=self.get_indices())

    def get_ohlc_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.get_ohlc(),
            index=self.get_indices(),
            columns=['Open', 'High', 'Low', 'Close']
        )

    def get_timeframe(self) -> str:
        if self._bar_method == 'time':
            secs = self._bar_size
            if secs < 60:
                return f'{secs}s'
            elif secs < 3600:
                return f'{secs // 60}min'
            else:
                return f'{secs // 3600}H'
        # for tick/volume/dollar bars, estimate average bar duration
        if self._indices is not None and len(self._indices) > 1:
            avg_delta = (self._indices[-1] -
                         self._indices[0]) / len(self._indices)
            minutes = avg_delta.total_seconds() / 60
            if minutes < 1:
                return '1min'
            return f'{int(round(minutes))}min'
        return '1min'

    def slice(self, start_date=None, end_date=None) -> 'TickData':
        if self._ohlc is None:
            raise RuntimeError("Call .aggregate() before slicing")

        start = start_date or self._indices[0]
        end = end_date or self._indices[-1]

        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)

        if start.tz is None:
            start = start.tz_localize(self._timezone)
        if end.tz is None:
            end = end.tz_localize(self._timezone)

        mask = (self._indices >= start) & (self._indices <= end)
        self._ohlc = self._ohlc[mask]
        self._indices = self._indices[mask]
        return self

    def split(self, split_date) -> Tuple['TickData', 'TickData']:
        if self._ohlc is None:
            raise RuntimeError("Call .aggregate() before splitting")

        if isinstance(split_date, str):
            split_date = pd.Timestamp(split_date)
        if split_date.tz is None:
            split_date = split_date.tz_localize(self._timezone)

        split_idx = self._indices.searchsorted(split_date)

        oos = TickData.__new__(TickData)
        oos._filepath = self._filepath
        oos._timezone = self._timezone
        oos._timestamps = None
        oos._prices = None
        oos._lows = None
        oos._highs = None
        oos._volumes = None
        oos._ohlc = self._ohlc[split_idx:]
        oos._indices = self._indices[split_idx:]
        oos._bar_method = self._bar_method
        oos._bar_size = self._bar_size

        self._ohlc = self._ohlc[:split_idx]
        self._indices = self._indices[:split_idx]

        return self, oos

    def get_raw_prices(self) -> np.ndarray:
        return self._prices

    def get_raw_timestamps(self) -> np.ndarray:
        return self._timestamps

    def get_raw_volumes(self) -> np.ndarray:
        return self._volumes

    def n_bars(self) -> int:
        if self._ohlc is None:
            return 0
        return len(self._ohlc)

    def n_raw_ticks(self) -> int:
        return len(self._prices)

    def __repr__(self) -> str:
        if self._ohlc is not None:
            return (
                f'TickData({self._bar_size} {self._bar_method} bars, '
                f'n_bars={self.n_bars():,}, '
                f'from {self._indices[0]} to {self._indices[-1]})'
            )
        return f'TickData(raw ticks={self.n_raw_ticks():,}, not yet aggregated)'
