import numpy as np
import pandas as pd


def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted features with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i + 1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def dynamic_window_rolling_average(x, attr):
    """
    Calculate rolling features with a dynamic window size for the specified attribute.

    Parameters:
        x (DataFrame): DataFrame containing the play-by-play data grouped by team.
        attr (str): The attribute for which rolling average is calculated.

    Returns:
        pd.Series: Series with the dynamic rolling EWMA for the attribute.
    """
    values = np.zeros(len(x))
    attr_shifted = f'{attr}_shifted'

    for i, (_, row) in enumerate(x.iterrows()):
        attr_data = x[attr_shifted][:i + 1]
        if row['week'] != 1:
            values[i] = attr_data.rolling(min_periods=1, window=row['week'] - 1).mean().values[-1]
        else:
            # Handle edge case for the first week or season start
            values[i] = attr_data.rolling(min_periods=1, window=18 if row['season'] >= 2021 else 17).mean().values[-1]

    return pd.Series(values, index=x.index)