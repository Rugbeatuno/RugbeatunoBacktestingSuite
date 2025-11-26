import numpy as np
import pandas as pd
import pandas_ta as ta
import re


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

    def __init__(self, filepath: str, symbol: str):
        """
        Parameters
        ----------
        filepath : str
            CSV file from DataBento containing multiple futures contracts.
        symbol : str
            symbol, e.g., "NQ", "ES", "CL".
        """

        self._filepath: str = filepath
        self.symbol = symbol.upper()
        self.symbol_len = len(symbol)
        self.timezone: str = None
        self.timeframe: str = '1min'
        self.valid_contract_pattern = f"^{self.symbol}[{self.MONTH_CODES}][0-9]$"

        self.df: pd.DataFrame = self._load_data()

    def __repr__(self) -> str:
        return self.df.to_string()

    def _load_data(self) -> pd.DataFrame:
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

        self.timezone = df.index.tz
        self.timeframe = '1min'

        assert df.index.is_unique, "Duplicate Indicies"

        print(f'\nStart:     {df.index[0]}')
        print(f'End:       {df.index[-1]}')
        print(f'Timezone:  {self.timezone}')
        print(f'Timeframe: {self.timeframe}')
        print(f"Loaded {len(df):,} clean rows\n")

        return df

    def convert_timezone(self, timezone: str):
        assert timezone in self.TIMEZONES, f"Invalid timezone '{timezone}'"

        self.timezone = timezone
        self.df.index = self.df.index.tz_convert(timezone)

    def find_gaps(self) -> pd.DatetimeIndex:
        start_time: pd.Timestamp = self.df.index[0]
        end_time: pd.Timestamp = self.df.index[-1]

        no_gaps: pd.DatetimeIndex = pd.date_range(
            start=start_time, end=end_time, freq=self.timeframe)
        missing: pd.DatetimeIndex = no_gaps.difference(self.df.index)

        return missing

    def get_resampled_data(self, timeframe: str) -> pd.DataFrame:
        assert timeframe in self.TIMEFRAMES, f'Invalid resampling timeframe: "{timeframe}"'

        self.df = self.df.resample(timeframe, label='right', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()

        self.timeframe = timeframe

        return self.df

    def get_close_prices(self, start: pd.Timestamp = None, end: pd.Timestamp = None) -> pd.Series:
        if start is not None and start not in self.df.index:
            raise ValueError(f"Start index {start} is not valid.")
        if end is not None and end not in self.df.index:
            raise ValueError(f"End index {end} is not valid.")

        if start is None and end is None:
            return self.df['Close']

        return self.df.loc[start:end, 'Close']

    def get_start_date(self) -> pd.Timestamp:
        return self.df.index[0]

    def get_end_date(self) -> pd.Timestamp:
        return self.df.index[-1]

    def get_raw_data(self) -> pd.DataFrame:
        return self.df

    def get_dates(self) -> pd.DatetimeIndex:
        return self.df.index

    def get_timeframe(self) -> str:
        return self.timeframe

    def generate_permuted_price_series(self) -> pd.Series:
        """
        Permutes returns (NOT prices) to create a randomized price path
        with identical return distribution but no market structure.
        """
        close = self.df['Close'].copy()
        log_returns = np.log(close).diff().fillna(0)

        # Shuffle returns
        shuffled_returns = np.random.permutation(log_returns.values)

        # Convert back to synthetic price series
        permuted_prices = pd.Series(
            close.iloc[0] * np.exp(np.cumsum(shuffled_returns)),
            index=log_returns.index
        )

        return permuted_prices

    def generate_permuted_ohlc_series(self) -> pd.DataFrame:
        """
        Generates a single permuted OHLC price series using shuffling and stitching.

        The process:
        1. Calculate bar ratios relative to the previous Close (inter-bar shift)
           and relative to the Open (intra-bar shape).
        2. Shuffle the vector of bar ratios.
        3. Reconstitute the price series by applying the shuffled ratios sequentially.

        Returns:
            A new pandas DataFrame with 'Open', 'High', 'Low', 'Close' in a 
            randomized time sequence, but maintaining bar integrity.
        """
        # --- 1. Calculate Bar Ratios ---
        # Calculate the inter-bar return (the shift/gap)
        # This is the Open of the current bar relative to the Close of the previous bar.
        prev_close = self.df['Close'].shift(1)
        # Use the first bar's Close as the starting point for the first Open calculation
        prev_close.iloc[0] = self.df['Close'].iloc[0]

        # Ratios that define the bar shape and shift
        ratios_df = pd.DataFrame(index=self.df.index)

        # R_shift: The shift from the previous close to the current open
        ratios_df['R_shift'] = self.df['Open'] / prev_close

        # Intra-bar ratios (defining the shape relative to the Open)
        ratios_df['R_high'] = self.df['High'] / self.df['Open']
        ratios_df['R_low'] = self.df['Low'] / self.df['Open']
        ratios_df['R_close'] = self.df['Close'] / self.df['Open']

        # Drop the first row which contains the initial Close price used for initialization
        ratios_to_shuffle = ratios_df.iloc[1:]

        # --- 2. Shuffle the Ratios (Permute the Bars) ---
        # Shuffle the rows of the ratio DataFrame. This destroys the temporal correlation
        # but keeps the OHLC-ratio structure of each bar intact.
        shuffled_indices = np.random.permutation(ratios_to_shuffle.index)
        shuffled_ratios = ratios_to_shuffle.loc[shuffled_indices].reset_index(
            drop=True)

        # --- 3. Reconstitute the Permuted Price Series (Stitching) ---
        permuted_prices = pd.DataFrame(index=self.df.index)

        # Initialize the series with the first actual bar's prices
        initial_bar = self.df.iloc[0]
        permuted_prices.loc[self.df.index[0]
                            ] = initial_bar[['Open', 'High', 'Low', 'Close']]

        # The initial price for the iterative calculation is the first bar's actual close
        current_close = initial_bar['Close']

        for i in range(len(shuffled_ratios)):
            # Get the permuted ratios for the current step (bar)
            bar_ratios = shuffled_ratios.iloc[i]

            # --- STITCHING: Calculate the current Open based on the previous Close ---
            # This applies the permuted shift (R_shift) to the last calculated Close price.
            permuted_open = current_close * bar_ratios['R_shift']

            # Calculate the rest of the OHLC bar based on its permuted shape (relative to the new Open)
            permuted_high = permuted_open * bar_ratios['R_high']
            permuted_low = permuted_open * bar_ratios['R_low']
            permuted_close = permuted_open * bar_ratios['R_close']

            # Store the new bar in the correct chronological spot (using the original index)
            # The index itself is just for labeling; the prices are temporally scrambled.
            current_index = self.df.index[i + 1]
            permuted_prices.loc[current_index] = [
                permuted_open, permuted_high, permuted_low, permuted_close
            ]

            # Update the starting point for the next iteration
            current_close = permuted_close

        # Round to avoid excessive floating point noise
        return permuted_prices.round(5)
