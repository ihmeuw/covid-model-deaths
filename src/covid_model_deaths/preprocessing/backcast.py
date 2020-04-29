import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS, LOCATIONS


def backcast_all_locations(df: pd.DataFrame, rate_threshold: float) -> pd.DataFrame:
    df = add_change_in_rate(df, COLUMNS.ln_age_death_rate, COLUMNS.delta_ln_asdr)
    df = add_change_in_rate(df, COLUMNS.obs_ln_age_death_rate, COLUMNS.observed_delta_ln_asdr)
    daily_change = get_backcast_daily_change_by_location(df, COLUMNS.delta_ln_asdr)

    bc_df = pd.concat([
        backcast_log_age_standardized_death_ratio(
            df.loc[df[COLUMNS.location_id] == bc_location_id],
            bc_location_id,
            daily_change.at[bc_location_id],
            rate_threshold,
        ) for bc_location_id in daily_change.index
    ])
    df = df.append(bc_df)
    df = df.sort_values([COLUMNS.location_id, COLUMNS.days]).reset_index(drop=True)
    # TODO: Document this assumption about back-filling.
    fill_cols = [COLUMNS.location, COLUMNS.country, COLUMNS.population]
    df[fill_cols] = df[fill_cols].fillna(method='backfill')
    df[COLUMNS.location_id] = df[COLUMNS.location_id].astype(int)
    df[COLUMNS.first_point] = df.groupby([COLUMNS.country, COLUMNS.location], as_index=False)[COLUMNS.days].transform('min')
    df.loc[df[COLUMNS.first_point] < 0, COLUMNS.days] = df[COLUMNS.days] - df[COLUMNS.first_point]
    del df[COLUMNS.first_point]
    return df


def add_change_in_rate(data: pd.DataFrame, measure: str, delta_measure: str) -> pd.DataFrame:
    """Compute and assign the daily difference in the measure."""
    required_columns = [COLUMNS.location_id, measure]
    assert set(required_columns).issubset(data.columns)
    data[delta_measure] = (
        data
        .groupby(COLUMNS.location_id)
        .apply(lambda x: x.set_index(COLUMNS.date)[measure] - x.set_index(COLUMNS.date)[measure].shift(1))
        .reset_index()[measure]
    )
    return data


def get_backcast_daily_change_by_location(data: pd.DataFrame, measure: str,
                                          average_window: int = 5, cutoff: float = 1e-4) -> pd.Series:
    """Compute the incremental change in the measure for the backcast.

    Parameters
    ----------
    data
        The data to compute the daily change for.
    measure
        The specific measure in the data to compute the daily change on.
    average_window
        The window size over which we wish to average in order to to get the
        daily change. The window used will start with the first day of data.
    cutoff
        The value below which we will not compute a daily change for back
        cast.

    Returns
    -------
        The daily change in the measure by location to use for back casting.

    """
    required_columns = [COLUMNS.location_id, COLUMNS.days, measure]
    assert set(required_columns).issubset(data.columns)
    # FIXME: Backcast should not care about this piece
    non_nursing_home = ~data[COLUMNS.location_id].isin([LOCATIONS.life_care.id])
    in_window = (0 < data[COLUMNS.days]) & (data[COLUMNS.days] <= average_window)
    daily_change = (data
                    .loc[in_window & non_nursing_home, [COLUMNS.location_id, measure]]
                    .groupby(COLUMNS.location_id)[measure]
                    .mean())
    daily_change = daily_change.loc[daily_change > cutoff]
    return daily_change


def backcast_log_age_standardized_death_ratio(df: pd.DataFrame, location_id: int,
                                              bc_step: int, rate_threshold: float) -> pd.DataFrame:
    """Backcast the log age standardized death rate back to the rate threshold."""
    out_columns = [COLUMNS.location_id, COLUMNS.days, COLUMNS.ln_age_death_rate]
    # get first point
    start_rep = df.sort_values(COLUMNS.days).reset_index(drop=True)[COLUMNS.ln_age_death_rate][0]

    if start_rep > rate_threshold:  # backcast
        # count from threshold on
        bc_rates = np.arange(rate_threshold, start_rep, bc_step)
        bc_df = pd.DataFrame({
            COLUMNS.location_id: location_id,
            COLUMNS.ln_age_death_rate: np.flip(bc_rates)
        })

        # remove fractional step from last (we force the threshold day to
        # be 0, so the partial day ends up getting added onto the first
        # day) no longer add date, since we have partial days
        if df[COLUMNS.days].min() != 0:
            raise ValueError(f'First day is not 0, as expected... (location_id: {location_id})')
        bc_df[COLUMNS.days] = -bc_df.index - (start_rep - bc_rates[-1]) / bc_step

        # don't project more than 10 days back, or we will have PROBLEMS
        bc_df = (bc_df
                 .loc[bc_df[COLUMNS.days] >= -10, out_columns]
                 .reset_index(drop=True))
    elif start_rep == rate_threshold:
        bc_df = pd.DataFrame(columns=out_columns)
    else:
        raise ValueError('First value is below threshold, should not be possible.')

    return bc_df
