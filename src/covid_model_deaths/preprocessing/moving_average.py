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

    if len(data) < window:
        return data
    
    # extend first two/last two diffs
    #data = np.exp(data)
    pre = data[0] - np.diff(data[:window]).mean()
    pre = pd.Series(pre, [data.index.min() - pd.Timedelta(days=1)], name=measure)
    pre.index.name = COLUMNS.date
    post = data[len(data)-1] + np.diff(data[len(data)-window:]).mean()
    post = pd.Series(post, [data.index.max() + pd.Timedelta(days=1)], name=measure)
    post.index.name = COLUMNS.date
    data = pd.concat([
        pre, data, post
    ])

    moving_average = (data.asfreq('D', method='pad')
                      .rolling(window=window, min_periods=1, center=True)
                      .mean())
    moving_average = moving_average[1:-1]
    #moving_average = np.log(moving_average)
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
    if len(data[COLUMNS.location_id].unique()) <= 1:
        moving_average = expanding_moving_average(data, measure, window).reset_index()
        moving_average[COLUMNS.location_id] = data[COLUMNS.location_id].unique()[0]
        moving_average = moving_average.set_index([COLUMNS.location_id, COLUMNS.date])
    else:
        moving_average = (
            data.loc[:, required_columns]
                .groupby(COLUMNS.location_id)
                .apply(lambda x: expanding_moving_average(x, measure, window))
        )
    return moving_average
