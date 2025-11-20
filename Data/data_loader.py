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
        self.timezone = 'America/Chicago'
        self.timeframe = '1min'

        self._filepath: str = filepath
        self.df: pd.DataFrame = self._load_data()

        # self._find_gaps()

    def __repr__(self):
        return self.df.to_string()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self._filepath, low_memory=True)

        print(f'\nLoading {self._filepath}')

        print(f'\nDataframe initial shape: {df.shape}')
        print(f'Dataframe initial columns: {df.columns}')

        df.index = df['Time']
        df.drop(columns=['Change', '%Chg', 'Time'],
                inplace=True, errors='raise')

        # sort the data
        df.sort_index(ascending=True, inplace=True)

        # remove duplicates
        df = df.groupby(df.index, sort=False).min()
        assert df.index.is_unique, "Duplicate Indicies!"

        # set index to datetime object
        df.index = pd.to_datetime(df.index)

        # have pandas figure out timezone
        df.index = df.index.tz_localize(self.timezone)

        print(f'\nDataframe post-processing shape: {df.shape}')
        print(f'Dataframe post-processing columns: {df.columns}')

        print(f'\nStart: {df.index[0]}')
        print(f'End: {df.index[-1]}')
        print(f'Timezone: {self.timezone}')

        print(f'\nTimeframe: {self.timeframe}')

        return df

    def convert_timezone(self, timezone: str):
        assert timezone in self.TIMEZONES, f"Invalid timezone '{timezone}'"

        self.df.index = self.df.index.tz_convert(timezone)

    def find_gaps(self) -> pd.DatetimeIndex:
        start_time: pd.Timestamp = self.df.index[0]
        end_time: pd.Timestamp = self.df.index[-1]

        no_gaps: pd.DatetimeIndex = pd.date_range(
            start=start_time, end=end_time, freq='1min')
        missing: pd.DatetimeIndex = no_gaps.difference(self.df.index)

        return missing

    def resample(self, timeframe: str) -> pd.DataFrame:
        assert timeframe in self.TIMEFRAMES, f'Invalid resampling timeframe: "{timeframe}"'

        resampled_df = self.df.resample(timeframe, label='right', closed='left').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()

        return resampled_df


if __name__ == '__main__':
    DataTimeseries('./data_combined.csv')
