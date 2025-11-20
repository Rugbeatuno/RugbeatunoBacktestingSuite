import numpy as np
import pandas as pd
import pandas_ta as ta


class DataTimeseries:

    TIMEFRAMES = (
        [str(m) + 'min' for m in range(1, 60)] +
        [str(h) + 'H' for h in range(1, 24)] +
        [str(d) + 'D' for d in range(1, 365)]
    )

    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.df: pd.DataFrame = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, low_memory=True)

        print(f'\nLoading {self.filepath}')

        print(f'\nDataframe initial shape: {df.shape}')
        print(f'Dataframe initial columns: {df.columns}')

        df.index = df['Time']
        df.drop(columns=['Change', '%Chg', 'Time'],
                inplace=True, errors='raise')

        print(f'\nDataframe post-processing shape: {df.shape}')
        print(f'Dataframe post-processing columns: {df.columns}')

        # remove duplicates indexes
        df = df.groupby(df.index).min()
        assert df.index.is_unique, "Duplicate Indicies!"

        # set index to datetime object
        df.index = pd.to_datetime(df.index)

        # sort the data
        df.sort_index(ascending=False, inplace=True)
        print(df)

        return df

    def resample(self, timeframe) -> pd.DataFrame:
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
