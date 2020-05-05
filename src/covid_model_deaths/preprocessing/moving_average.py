"""Tools for averaging data over time."""
import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS


def expanding_moving_average(data: pd.DataFrame, measure: str, window: int) -> pd.Series:
    """Expands a dataset over date and performs a moving average.

    Parameters
    ----------
    data
        The dataset to perform the moving average over.
    measure
        The column name in the dataset to average.
    window
        The number of days to average over.

    Returns
    -------
        A series indexed by the expanded date with the measure averaged
        over the window.

    """
    required_columns = [COLUMNS.date, measure]
    data = data.loc[:, required_columns].set_index(COLUMNS.date).loc[:, measure]

    #if len(data) < 6:
    if len(data) < window:
        return data

    moving_average = (data
                      .asfreq('D', method='pad')
                      .rolling(window=window, min_periods=1, center=True)
                      .mean())

    # # # project avg last two second derivative forward at end
    # # first_diff = np.diff(moving_average[-5:-1])
    # # second_diff = np.diff(first_diff).mean()
    # # last_step = first_diff[-1] + second_diff
    # project last derivative forward at end
    last_step = np.diff(moving_average[-window:-1]) # np.mean(np.diff(moving_average[-window - 1:-1]))
    moving_average.iloc[-1] = moving_average.iloc[-2] + last_step

    # # # project avg first two second derivative backward at beginning
    # # first_diff = np.diff(moving_average[1:5])
    # # second_diff = np.diff(first_diff).mean()
    # # first_step = first_diff[0] - second_diff
    # project first derivative forward at beginning
    first_step = np.mean(np.diff(moving_average[1:window]))
    moving_average.iloc[0] = moving_average.iloc[1] - first_step
    return moving_average


def expanding_moving_average_by_location(data: pd.DataFrame, measure: str, window: int = 3) -> pd.Series:
    """Expands a dataset over date and performs a moving average by location.

    Parameters
    ----------
    data
        The dataset to perform the moving average over.
    measure
        The column name in the dataset to average.
    window
        The number of days to average over.

    Returns
    -------
        A dataframe indexed by location id and the expanded date with the
        measure averaged over the window.

    """
    required_columns = [COLUMNS.location_id, COLUMNS.date, measure]
    moving_average = (
        data.loc[:, required_columns]
            .groupby(COLUMNS.location_id)
            .apply(lambda x: expanding_moving_average(x, measure, window))
    )
    return moving_average
