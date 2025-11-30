import numpy as np
import pandas as pd
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
        self.symbol: str = symbol.upper()
        self.symbol_len: int = len(symbol)
        self.timezone: str = None
        self.timeframe: str = '1min'
        self.valid_contract_pattern = f"^{self.symbol}[{self.MONTH_CODES}][0-9]$"

        self.indicies: pd.DatetimeIndex = None
        self.ohlc: np.ndarray = None
        self.volume: np.ndarray = None

        self._load_data()

    def __repr__(self) -> str:
        return self.get_dataframe().head().to_string()

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

        self.timezone = df.index.tz
        self.timeframe = '1min'

        assert df.index.is_unique, "Duplicate Indicies"

        print(f'\nStart:     {df.index[0]}')
        print(f'End:       {df.index[-1]}')
        print(f'Timezone:  {self.timezone}')
        print(f'Timeframe: {self.timeframe}')
        print(f"Loaded {len(df):,} clean rows\n")

        self.indicies = df.index
        self.ohlc = df[['Open', 'High', 'Low', 'Close']].values
        self.volume = df[['Volume']].values

    def convert_timezone(self, timezone: str):
        assert timezone in self.TIMEZONES, f"Invalid timezone '{timezone}'"

        self.timezone = timezone
        self.indicies = self.indicies.tz_convert(timezone)

    def find_gaps(self) -> pd.DatetimeIndex:
        # start_time: pd.Timestamp = self.df.index[0]
        # end_time: pd.Timestamp = self.df.index[-1]

        # no_gaps: pd.DatetimeIndex = pd.date_range(
        #     start=start_time, end=end_time, freq=self.timeframe)
        # missing: pd.DatetimeIndex = no_gaps.difference(self.df.index)

        pass

    def resample_data(self, timeframe: str) -> None:
        assert timeframe in self.TIMEFRAMES, f'Invalid resampling timeframe: "{timeframe}"'

        # --- 1. CONVERT BACK TO TEMPORARY PANDAS DATAFRAME ---
        # The index (self.indicies) is necessary for resampling
        temp_df = self.get_dataframe()
        temp_df = temp_df.resample(timeframe, label='right', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()

        self.timeframe = timeframe
        self.indicies = temp_df.index
        self.ohlc = temp_df[['Open', 'High', 'Low', 'Close']].values
        self.volume = temp_df[['Volume']].values

    def get_indicies(self) -> pd.DatetimeIndex:
        return self.indicies

    def get_close_prices(self) -> np.ndarray:
        return self.ohlc[:, 3]

    def get_close_prices_series(self) -> pd.Series:
        return pd.Series(self.get_close_prices(), index=self.indicies)

    def get_ohlc_prices(self) -> np.ndarray:
        return self.ohlc

    def get_start_date(self) -> pd.Timestamp:
        return self.indicies[0]

    def get_end_date(self) -> pd.Timestamp:
        return self.indicies[-1]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.ohlc,
            index=self.indicies,
            columns=['Open', 'High', 'Low', 'Close']
        )
        df['Volume'] = self.volume

        return df

    def get_timeframe(self) -> str:
        return self.timeframe

    @staticmethod
    def generate_permuted_close_prices(close_prices: np.ndarray) -> np.ndarray:
        """
        Permutes returns (NOT prices) to create a randomized price path
        with identical return distribution but no market structure.
        """
        log_returns = np.diff(np.log(close_prices))
        log_returns_padded = np.insert(log_returns, 0, 0.0)
        shuffled_returns = np.random.permutation(log_returns_padded)

        # Convert back to synthetic price series
        rescaled_returns = close_prices[0] * \
            np.exp(np.cumsum(shuffled_returns))

        return rescaled_returns

    @staticmethod
    def generate_permuted_ohlc_prices(ohlc: np.ndarray) -> np.ndarray:
        """
        Generates a single permuted OHLC price series using shuffling and stitching.

        The process:
        1. Calculate bar ratios relative to the previous Close (inter-bar shift)
           and relative to the Open (intra-bar shape).
        2. Shuffle the vector of bar ratios.
        3. Reconstitute the price series by applying the shuffled ratios sequentially.

        Returns:
            A new numpy array with 'Open', 'High', 'Low', 'Close' in a
            randomized time sequence, but maintaining bar integrity.
        """
        # --- 1. Calculate Bar Ratios ---
        # Calculate the inter-bar return (the shift/gap)
        # This is the Open of the current bar relative to the Close of the previous bar.
        prev_close = np.roll(ohlc[:, 3], 1)
        prev_close[0] = ohlc[0, 3]  # first bar uses its own Close

        # Ratios that define the bar shape and shift
        R_shift = ohlc[:, 0] / prev_close
        R_high = ohlc[:, 1] / ohlc[:, 0]
        R_low = ohlc[:, 2] / ohlc[:, 0]
        R_close = ohlc[:, 3] / ohlc[:, 0]

        ratios = np.column_stack((R_shift, R_high, R_low, R_close))

        # --- 2. Shuffle the Ratios (Permute the Bars) ---
        shuffled_indices = np.random.permutation(ratios.shape[0])
        shuffled_ratios = ratios[shuffled_indices]

        # --- 3. Reconstitute the Permuted Price Series (Stitching) ---
        permuted_prices = np.zeros_like(ohlc)

        # Initialize the series with the first actual bar's prices
        permuted_prices[0] = ohlc[0]

        current_close = ohlc[0, 3]

        for i in range(1, len(shuffled_ratios)):
            bar_ratios = shuffled_ratios[i]

            permuted_open = current_close * bar_ratios[0]
            permuted_high = permuted_open * bar_ratios[1]
            permuted_low = permuted_open * bar_ratios[2]

    @staticmethod
    def generate_permuted_close_series(close_prices: np.ndarray, indicies: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(DataTimeseries.generate_permuted_close_prices(close_prices), index=indicies)

    @staticmethod
    def permute_cme_ohlc(ohlc: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Permute OHLC candles according to Timothy Masters' rules:
        - Preserve weekend gaps (do not permute gap candles)
        - Separate RTH vs ETH sessions
        - Permute only candle-shape vectors
        - Reconstruct synthetic prices using permuted shapes

        Parameters
        ----------
        ohlc : np.ndarray
            Array shaped (N, 4) with columns [open, high, low, close]
        timestamps : pd.DatetimeIndex
            N-length array of pandas DatetimeIndex timestamps.

        Returns
        -------
        np.ndarray
            Synthetic OHLC array with same shape.
        """

        # ----- 1. Identify sessions -----
        # CME ES/NQ/etc: RTH = 09:30–16:00 ET, ETH = everything outside that
        hours = timestamps.hour % 24
        minutes = timestamps.minute % 60

        # RTH mask 09:30–16:00
        rth_mask = (hours > 9) & (hours < 16) | (
            (hours == 9) & (minutes >= 30))
        rth_mask |= (hours == 16) & (minutes == 0)    # allow 16:00 exact close

        # ETH mask = inverse
        eth_mask = ~rth_mask

        # ----- 2. Detect weekend gaps -----
        # any time jump > 4 hours = weekend/holiday gap
        deltas = timestamps.to_series().diff().dt.total_seconds().fillna(0)
        gap_mask = (deltas > (4 * 3600)).values

        # We DO NOT permute gap bars.
        permute_mask = ~gap_mask

        # ----- 3. Build candle-shape vectors -----
        # Shapes are relative moves within each bar
        open_ = ohlc[:, 0]
        high_ = ohlc[:, 1]
        low_ = ohlc[:, 2]
        close_ = ohlc[:, 3]

        # Shape vector is [High - Open, Low - Open, Close - Open]
        shape = np.column_stack([
            (high_ - open_),
            (low_ - open_),
            (close_ - open_)
        ])

        # ----- 4. Split shapes by session type and permute status -----
        shape_rth = shape[rth_mask & permute_mask]
        shape_eth = shape[eth_mask & permute_mask]

        # ----- 5. Permute shapes within each session -----
        idx_rth = np.random.permutation(len(shape_rth))
        idx_eth = np.random.permutation(len(shape_eth))

        perm_rth = shape_rth[idx_rth]
        perm_eth = shape_eth[idx_eth]

        # ----- 6. Reassemble shape array in the original order -----
        perm_shape = np.zeros_like(shape)

        # Place permuted shapes back into their session slots
        perm_shape[rth_mask & permute_mask] = perm_rth
        perm_shape[eth_mask & permute_mask] = perm_eth

        # Weekend/holiday gap candles keep their shape unpermuted
        perm_shape[gap_mask] = shape[gap_mask]

        # ----- 7. Reconstruct synthetic OHLC ----------
        synthetic = np.zeros_like(ohlc, dtype=float)

        # First synthetic open = original open
        synthetic[0, 0] = open_[0]

        for i in range(len(ohlc)):
            if i == 0:
                # Calculate the first bar's OHLC from its open and permuted shape
                o = synthetic[i, 0]
                sh = perm_shape[i]
            else:
                # --- FIX APPLIED HERE ---
                # We need to preserve the Close-to-Open return (the overnight/session gap)
                # The dollar return is: Original Open[i] - Original Close[i-1]
                close_to_open_return = open_[i] - close_[i - 1]

                # New synthetic open = Previous synthetic close + original C/O return
                synthetic[i, 0] = synthetic[i - 1, 3] + close_to_open_return
                # --- END FIX ---

                o = synthetic[i, 0]
                sh = perm_shape[i]

            synthetic[i, 1] = o + sh[0]  # high
            synthetic[i, 2] = o + sh[1]  # low
            synthetic[i, 3] = o + sh[2]  # close

            # Since the original data maintains O <= L and O >= H, and the shape is preserved,
            # we should technically also maintain the min/max relationships.
            # However, for robustness, we explicitly ensure H is the max and L is the min.
            synthetic[i, 1] = max(
                synthetic[i, 1], synthetic[i, 0], synthetic[i, 3])
            synthetic[i, 2] = min(
                synthetic[i, 2], synthetic[i, 0], synthetic[i, 3])

        return synthetic

    # @staticmethod
    # def get_permutation(ohlc: np.ndarray, indicies: pd.DatetimeIndex) -> np.ndarray:

    #     # --- 1. Initial Checks and Setup ---
    #     log_ohlc = np.log(ohlc)
    #     N = log_ohlc.shape[0]
    #     start_idx: int = 1
    #     permute_size = N - start_idx

    #     # --- 2. Define RTH/ETH Bar Segments (For Intra-Bar Shuffling) ---

    #     # Define RTH (Regular Trading Hours) window (e.g., 9:30 to 16:00)
    #     RTH_START_MINUTES = 9 * 60 + 30  # 9:30 AM
    #     RTH_END_MINUTES = 16 * 60        # 4:00 PM (16:00)

    #     all_times_minutes = indicies.hour * 60 + indicies.minute

    #     # RTH Mask: True for bars that fall within Regular Trading Hours
    #     rth_bar_mask = (all_times_minutes >= RTH_START_MINUTES) & \
    #                    (all_times_minutes <= RTH_END_MINUTES)
    #     # ETH Mask: True for all other bars
    #     eth_bar_mask = ~rth_bar_mask

    #     # --- 3. Decompose Prices into Relative Changes (Log Returns) ---
    #     o_t = log_ohlc[start_idx:, 0]
    #     h_t = log_ohlc[start_idx:, 1]
    #     l_t = log_ohlc[start_idx:, 2]
    #     c_t = log_ohlc[start_idx:, 3]
    #     c_prev = log_ohlc[start_idx - 1:-1, 3]

    #     # Close-to-Open Gap (The component we will segment)
    #     rel_open = o_t - c_prev

    #     # Intra-bar changes
    #     rel_high = h_t - o_t
    #     rel_low = l_t - o_t
    #     rel_close = c_t - o_t

    #     # --- 4. Identify and Segment Gap Types (For Inter-Bar Shuffling) ---

    #     close_times = indicies[start_idx - 1: N - 1]
    #     open_times = indicies[start_idx: N]
    #     gap_duration = (open_times - close_times).total_seconds() / 3600.0

    #     # A. Weekend/Holiday Gap (WG) Mask: Duration >= 4 hours (long break)
    #     WG_THRESHOLD_HOURS = 4.0
    #     wg_mask = (gap_duration >= WG_THRESHOLD_HOURS)

    #     # B. Short Duration Mask: All gaps not classified as WG
    #     short_duration_mask = ~wg_mask

    #     # Get the bar masks sliced to match the size of the permutable changes (N-1)
    #     rth_bar_mask_permute = rth_bar_mask[start_idx:]
    #     eth_bar_mask_permute = eth_bar_mask[start_idx:]

    #     # C. RTH Gap Mask: A short gap that precedes an RTH bar
    #     rth_gap_mask = short_duration_mask & rth_bar_mask_permute

    #     # D. ETH Gap Mask: A short gap that precedes an ETH bar
    #     eth_gap_mask = short_duration_mask & eth_bar_mask_permute

    #     # Sanity check: All masks should cover every gap exactly once.
    #     if not np.all(rth_gap_mask | eth_gap_mask | wg_mask):
    #         raise ValueError(
    #             "Gap segmentation masks do not cover all data points.")

    #     # --- 5. Apply Segmented Permutation ---

    #     # --- 5a. Permute Intra-Bar Movements (RTH shuffled with RTH, ETH with ETH) ---
    #     rel_high_shuffled = np.empty_like(rel_high)
    #     rel_low_shuffled = np.empty_like(rel_low)
    #     rel_close_shuffled = np.empty_like(rel_close)

    #     # RTH Intra-Bar Shuffle
    #     rth_high_segment = rel_high[rth_bar_mask_permute]
    #     rth_low_segment = rel_low[rth_bar_mask_permute]
    #     rth_close_segment = rel_close[rth_bar_mask_permute]

    #     # Create and shuffle the RTH permutation indices
    #     p_rth = np.arange(len(rth_high_segment))
    #     np.random.shuffle(p_rth)

    #     rel_high_shuffled[rth_bar_mask_permute] = rth_high_segment[p_rth]
    #     rel_low_shuffled[rth_bar_mask_permute] = rth_low_segment[p_rth]
    #     rel_close_shuffled[rth_bar_mask_permute] = rth_close_segment[p_rth]

    #     # ETH Intra-Bar Shuffle
    #     eth_high_segment = rel_high[eth_bar_mask_permute]
    #     eth_low_segment = rel_low[eth_bar_mask_permute]
    #     eth_close_segment = rel_close[eth_bar_mask_permute]

    #     # Create and shuffle the ETH permutation indices
    #     p_eth = np.arange(len(eth_high_segment))
    #     np.random.shuffle(p_eth)

    #     rel_high_shuffled[eth_bar_mask_permute] = eth_high_segment[p_eth]
    #     rel_low_shuffled[eth_bar_mask_permute] = eth_low_segment[p_eth]
    #     rel_close_shuffled[eth_bar_mask_permute] = eth_close_segment[p_eth]

    #     # --- 5b. Segment and Permute the Gap pools (RTH Gap, ETH Gap, Weekend Gap) ---
    #     rel_open_shuffled = np.empty_like(rel_open)

    #     # RTH Gap Shuffle
    #     rth_gaps = rel_open[rth_gap_mask]
    #     np.random.shuffle(rth_gaps)
    #     rel_open_shuffled[rth_gap_mask] = rth_gaps

    #     # ETH Gap Shuffle
    #     eth_gaps = rel_open[eth_gap_mask]
    #     np.random.shuffle(eth_gaps)
    #     rel_open_shuffled[eth_gap_mask] = eth_gaps

    #     # Weekend/Holiday Gap Shuffle
    #     wg_gaps = rel_open[wg_mask]
    #     np.random.shuffle(wg_gaps)
    #     rel_open_shuffled[wg_mask] = wg_gaps

    #     print(permute_size, rth_bar_mask.sum(), eth_bar_mask.sum(),
    #           rth_gap_mask.sum(), eth_gap_mask.sum(), wg_mask.sum())

    #     # --- 6. Reconstruct the Prices ---

    #     permuted_log_ohlc = np.zeros_like(log_ohlc)
    #     permuted_log_ohlc[0, :] = log_ohlc[0, :]
    #     prev_close = permuted_log_ohlc[0, 3]

    #     print(prev_close.max())
    #     for i in range(permute_size):
    #         gap = rel_open_shuffled[i]
    #         r_h = rel_high_shuffled[i]
    #         r_l = rel_low_shuffled[i]
    #         r_c = rel_close_shuffled[i]

    #         # Reconstruction based on permuted log returns
    #         new_open = prev_close + gap
    #         new_high = new_open + r_h
    #         new_low = new_open + r_l
    #         new_close = new_open + r_c

    #         permuted_log_ohlc[i + start_idx, 0] = new_open
    #         permuted_log_ohlc[i + start_idx, 1] = new_high
    #         permuted_log_ohlc[i + start_idx, 2] = new_low
    #         permuted_log_ohlc[i + start_idx, 3] = new_close

    #         prev_close = new_close

    #     # --- 7. Convert Back to Actual Prices ---
    #     return np.exp(permuted_log_ohlc)

    @staticmethod
    def get_permutation(ohlc: np.ndarray, indicies: pd.DatetimeIndex) -> np.ndarray:
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
