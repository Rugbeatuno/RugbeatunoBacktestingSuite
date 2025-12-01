import numpy as np
import pandas as pd
import re
from typing import Tuple


class DataTimeseries:
    """
    Loader for DataBento futures data.
    Automatically:
        - Cleans timestamped OHLCV data
        - Identifies valid contracts for a root symbol (e.g., NQ)
        - Rejects DataBento's synthetic bridge contracts (e.g., NQM3-NQU3)
        - Builds a continuous back-adjusted futures series
        - Keeps the originating contract as a column
    """
    TIMEFRAMES = (
        [str(m) + 'min' for m in range(1, 60)] +
        [str(h) + 'H' for h in range(1, 24)] +
        [str(d) + 'D' for d in range(1, 365)]
    )

    TIMEZONES = [
        'UTC',
        'America/New_York',
        'America/Chicago',
        'America/Los_Angeles',
        'Europe/London',
        'Europe/Berlin',
        'Europe/Paris',
        'Europe/Madrid',
        'Europe/Rome',
        'Europe/Moscow',
        'Asia/Taipei',
        'Asia/Tokyo',
        'Asia/Shanghai',
        'Asia/Hong_Kong',
        'Asia/Singapore',
        'Asia/Seoul'
    ]

    MONTH_CODES = "FGHJKMNQUVXZ"   # standard CME month codes

    def __init__(self, filepath: str = None):
        """
        Parameters
        ----------
        filepath : str
            CSV file from DataBento containing multiple futures contracts.
        symbol : str
            symbol, e.g., "NQ", "ES", "CL".
        """

        if filepath is None:
            return

        self._filepath: str = filepath
        self._timezone: str = None
        self._timeframe: str = '1min'

        self._indicies: pd.DatetimeIndex = None
        self._ohlc: np.ndarray = None
        self._volume: np.ndarray = None

        self._load_data()

    def __repr__(self) -> str:
        return self.get_close_prices_series().head().to_string()

    def _load_data(self) -> None:
        print(f'\nLoading {self._filepath}')
        df: pd.DataFrame = pd.read_csv(self._filepath, low_memory=True)

        # required columns
        required = ["Time", "Open", "High",
                    "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: {}".format(missing))

        df = df[required]  # remove non-required columns
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        # set index to time column, and drop time column
        df.set_index(df['Time'], drop=True, inplace=True)
        df.sort_index(ascending=True, inplace=True)

        # Attach timezone if missing
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        self._timezone = df.index.tz
        self._timeframe = '1min'

        assert df.index.is_unique, "Duplicate Indicies"

        print(f'\nStart:     {df.index[0]}')
        print(f'End:       {df.index[-1]}')
        print(f'Timezone:  {self._timezone}')
        print(f'Timeframe: {self._timeframe}')
        print(f"Loaded {len(df):,} clean rows\n")

        self._indicies = df.index
        self._ohlc = df[['Open', 'High', 'Low', 'Close']].values
        self._volume = df[['Volume']].values

    def convert_timezone(self, timezone: str):
        assert timezone in self.TIMEZONES, f"Invalid timezone '{timezone}'"

        self._timezone = timezone
        self._indicies = self._indicies.tz_convert(timezone)

    def resample_data(self, timeframe: str) -> None:
        assert timeframe in self.TIMEFRAMES, f'Invalid resampling timeframe: "{timeframe}"'

        # --- 1. CONVERT BACK TO TEMPORARY PANDAS DATAFRAME ---
        # The index (self.indicies) is necessary for resampling
        temp_df = self.get_close_prices_series()
        temp_df = temp_df.resample(timeframe, label='right', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()

        self._timeframe = timeframe
        self._indicies = temp_df.index
        self._ohlc = temp_df[['Open', 'High', 'Low', 'Close']].values
        self._volume = temp_df[['Volume']].values

    def split(self, split_date: pd.Timestamp) -> Tuple['DataTimeseries', 'DataTimeseries']:
        # split the currently loaded data
        split_idx = self._indicies.searchsorted(split_date)

        oos = DataTimeseries()
        oos._indicies = self._indicies[split_idx:]
        oos._ohlc = self._ohlc[split_idx:]
        oos._volume = self._volume[split_idx:]
        oos._timeframe = self._timeframe
        oos._timezone = self._timezone
        oos._filepath = self._filepath

        self._indicies = self._indicies[:split_idx]
        self._ohlc = self._ohlc[:split_idx]
        self._volume = self._volume[:split_idx]

        return self, oos

    def get_indicies(self) -> pd.DatetimeIndex:
        return self._indicies

    def get_close_prices(self) -> np.ndarray:
        return self._ohlc[:, 3]

    def get_close_prices_series(self) -> pd.Series:
        return pd.Series(self.get_close_prices(), index=self._indicies)

    def get_ohlc_prices(self) -> np.ndarray:
        return self._ohlc

    def get_start_date(self) -> pd.Timestamp:
        return self._indicies[0]

    def get_end_date(self) -> pd.Timestamp:
        return self._indicies[-1]

    def get_ohlc_prices_series(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self._ohlc,
            index=self._indicies,
            columns=['Open', 'High', 'Low', 'Close']
        )
        df['Volume'] = self._volume

        return df

    def get_timeframe(self) -> str:
        return self._timeframe

    def get_permutation(self) -> np.ndarray:
        return DataTimeseries.generate_permutation(self._ohlc, self._indicies)

    @staticmethod
    def generate_permutation(ohlc: np.ndarray, indicies: pd.DatetimeIndex) -> np.ndarray:
        assert indicies is not None

        if indicies.tz is None:
            indicies = indicies.tz_localize(
                "UTC").tz_convert("America/New_York")
        else:
            indicies = indicies.tz_convert("America/New_York")

        n_bars: int = ohlc.shape[0]
        start_index: int = 1
        permute_size: int = n_bars - start_index

        log_bars: np.ndarray = np.log(ohlc)

        # assign start bar
        start_bar = log_bars[start_index - 1, :]

        # open relative to last close
        # get prices relative to this bars open
        r_o = log_bars[:, 0] - np.roll(log_bars[:, 3], shift=1)
        r_h = log_bars[:, 1] - log_bars[:, 0]
        r_l = log_bars[:, 2] - log_bars[:, 0]
        r_c = log_bars[:, 3] - log_bars[:, 0]

        # align with permute_size
        r_o = r_o[start_index:]
        r_h = r_h[start_index:]
        r_l = r_l[start_index:]
        r_c = r_c[start_index:]
        indicies = indicies[start_index:]

        # ----- 1. Identify sessions -----
        # CME ES/NQ/etc: RTH = 09:30–16:00 ET, ETH = everything outside that
        # Identify weekends first, only permute weekend gaps amongst themselves
        # then identify RTH/ETH within the remaining bars
        # only permute RTH with RTH and ETH with ETH

        # any time jump > 4 hours = weekend/holiday gap
        deltas = indicies.to_series().diff().dt.total_seconds().fillna(0)
        gap_mask = (deltas > (24 * 3600)).values

        hours = indicies.hour % 24
        minutes = indicies.minute % 60

        # RTH mask 09:30–16:00
        rth_mask = (hours > 9) & (hours < 16) | (
            (hours == 9) & (minutes >= 30))
        # rth_mask |= (hours == 16) & (minutes == 0)    # allow 16:00 exact close

        rth_mask = rth_mask & ~gap_mask
        eth_mask = ~rth_mask & ~gap_mask

        assert gap_mask.sum() + rth_mask.sum() + eth_mask.sum() == permute_size
        # ----- 2. Permute within each group -----
        idx = np.arange(permute_size)

        # Weekend gaps
        # shuffle intrabar relative values (high, low, close)
        weekend_gaps = idx[gap_mask]
        perm1 = np.random.permutation(weekend_gaps)
        r_h[gap_mask] = r_h[perm1]
        r_l[gap_mask] = r_l[perm1]
        r_c[gap_mask] = r_c[perm1]
        # shuffle last close to open (gaps) separately
        perm2 = np.random.permutation(weekend_gaps)
        r_o[gap_mask] = r_o[perm2]

        # RTH bars
        rth_bars = idx[rth_mask]
        perm3 = np.random.permutation(rth_bars)
        r_h[rth_mask] = r_h[perm3]
        r_l[rth_mask] = r_l[perm3]
        r_c[rth_mask] = r_c[perm3]

        perm4 = np.random.permutation(rth_bars)
        r_o[rth_mask] = r_o[perm4]

        # ETH bars
        eth_bars = idx[eth_mask]
        perm5 = np.random.permutation(eth_bars)
        r_h[eth_mask] = r_h[perm5]
        r_l[eth_mask] = r_l[perm5]
        r_c[eth_mask] = r_c[perm5]

        perm6 = np.random.permutation(eth_bars)
        r_o[eth_mask] = r_o[perm6]

        # ----- 3. Reconstruct prices -----
        start_close = start_bar[3]

        permuted_log_close = start_close + np.cumsum(r_o + r_c)
        permuted_log_open = permuted_log_close - r_c
        permuted_log_high = permuted_log_open + r_h
        permuted_log_low = permuted_log_open + r_l

        permuted_log_ohlc = np.stack([
            permuted_log_open,
            permuted_log_high,
            permuted_log_low,
            permuted_log_close
        ], axis=1)

        final_permuted_log_ohlc = np.zeros_like(ohlc, dtype=float)
        final_permuted_log_ohlc[0, :] = start_bar
        final_permuted_log_ohlc[1:, :] = permuted_log_ohlc

        assert final_permuted_log_ohlc.shape == ohlc.shape

        return np.exp(final_permuted_log_ohlc)
