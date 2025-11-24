import numpy as np
import pandas as pd
import pandas_ta as ta


class DataTimeseries:

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

    def __init__(self, filepath: str):
        self.timezone: str = 'America/Chicago'
        self.timeframe = '1min'

        self._filepath: str = filepath
        self.df: pd.DataFrame = self._load_data()

        # self._find_gaps()

    def __repr__(self) -> str:
        return self.df.to_string()

    def _load_data(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self._filepath, low_memory=True)

        print(f'\nLoading {self._filepath}')

        print(f'\nDataframe initial shape: {df.shape}')
        print(f'Dataframe initial columns: {df.columns}')

        df.index = df['Time']
        df.drop(
            columns=['Change', '%Chg', 'Time', 'ts_event', 'rtype',
                     'publisher_id', 'instrument_id', 'symbol'],
            inplace=True,
            errors='ignore'
        )

        # sort the data
        df.sort_index(ascending=True, inplace=True)

        # remove duplicates
        df = df.groupby(df.index, sort=False).max()
        assert df.index.is_unique, "Duplicate Indicies!"

        # drop zeros or any value less than 1000
        df = df[(df[['Open', 'High', 'Low', 'Close']] > 1000).all(axis=1)]

        # set index to datetime object
        df.index = pd.to_datetime(df.index)
        self.timezone = df.index.tz

        # have pandas figure out timezone
        # df.index = df.index.tz_localize(self.timezone)

        print(f'\nDataframe post-processing shape: {df.shape}')
        print(f'Dataframe post-processing columns: {df.columns}')

        print(f'\nStart: {df.index[0]}')
        print(f'End: {df.index[-1]}')
        print(f'Timezone: {self.timezone}')

        print(f'\nTimeframe: {self.timeframe}')

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
