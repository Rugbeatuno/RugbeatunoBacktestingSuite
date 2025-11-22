
import pandas as pd
import numpy as np

signal = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0])
print(signal)
print(signal[signal.ne(signal.shift())])
