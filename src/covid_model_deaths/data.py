import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS, LOCATIONS


def process_death_df(death_df: pd.DataFrame, subnat: bool) -> pd.DataFrame:
    """Subset and filter the death data."""
    death_df = death_df.sort_values([COLUMNS.country, COLUMNS.location, COLUMNS.date]).reset_index(drop=True)

    if subnat:
        # FIXME: Faulty logic.  Use location hierarchy
        location_matches_country = death_df[COLUMNS.location] == death_df[COLUMNS.country]
        mexico_is_special = death_df[COLUMNS.location_id] == 4657
        death_df = death_df.loc[~location_matches_country | mexico_is_special].reset_index(drop=True)

    bad_locations = ['Outside Wuhan City, Hubei', 'Outside Hubei']
    bad_location_data = death_df[COLUMNS.location].isin(bad_locations)
    death_df = death_df.loc[~bad_location_data].reset_index(drop=True)

    # TODO: Check preconditions on data sets well before this.
    n_loc_ids = len(death_df[[COLUMNS.location_id]].drop_duplicates())
    n_country_locations = len(death_df[[COLUMNS.country, COLUMNS.location]].drop_duplicates())
    n_id_country_locations = len(death_df[[COLUMNS.location_id, COLUMNS.country, COLUMNS.location]].drop_duplicates())
    if not n_loc_ids == n_country_locations == n_id_country_locations:
        raise ValueError(
            'location_id, Country/Region + Location, and location_id + Country/Region + Location '
            'are not all acceptable keys. I assume this is true, check why not.'
        )
    return death_df


def get_standard_age_death_df(age_death_df: pd.DataFrame) -> pd.DataFrame:
    """Compute or select the age death pattern."""
    global_loc = age_death_df[COLUMNS.location_id] == LOCATIONS.global_aggregate.id
    keep_columns = [COLUMNS.age_group, COLUMNS.death_rate_bad]
    return age_death_df.loc[global_loc, keep_columns].reset_index(drop=True)


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


def drop_lagged_deaths_by_location(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows from data where no new deaths occurred on the current day.

    Parameters
    ----------
    data
        The dataset containing lagged deaths.

    Returns
    -------
        The same dataset with rows containing lagged deaths removed.

    """
    required_columns = [COLUMNS.location_id, COLUMNS.date, COLUMNS.deaths]
    lagged_deaths = (
        data.loc[:, required_columns]
            .groupby(COLUMNS.location_id, as_index=False)
            .apply(lambda x: ((x[COLUMNS.date] == x[COLUMNS.date].max())
                              & (x[COLUMNS.deaths] == x[COLUMNS.deaths].shift(1))))
            .droplevel(0)
    )
    return data.loc[~lagged_deaths]


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

    moving_average = (data
                      .asfreq('D', method='pad')
                      .rolling(window=window, min_periods=1, center=True)
                      .mean())
    if len(moving_average) > window:
        # replace last point w/ daily value over 3->2 and 2->1 and the first
        # with 1->2, 2->3; use observed if 3 data points or less
        last_step = np.mean(np.array(moving_average[-window:-1]) - np.array(moving_average[-window - 1:-2]))
        moving_average.iloc[-1] = moving_average.iloc[-2] + last_step

        first_step = np.mean(np.array(moving_average[2:window + 1]) - np.array(moving_average[1:window]))
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


def add_moving_average_ln_asdr(data: pd.DataFrame, rate_threshold: float) -> pd.DataFrame:
    """Smooths over the log age specific death rate.

    Parameters
    ----------
    data
        The data with the age specific death rate to smooth over.
    rate_threshold
        The minimum age specific death rate.  Values produced in the
        averaging will be pinned to this.

    Returns
    -------
        The same data with the log asdr replaced with its average and a new
        column with the original observed asdr.

    """
    data[COLUMNS.obs_ln_age_death_rate] = data[COLUMNS.ln_age_death_rate]
    moving_average = expanding_moving_average_by_location(data, COLUMNS.ln_age_death_rate)
    # noinspection PyTypeChecker
    moving_average[moving_average < rate_threshold] = rate_threshold
    data = data.set_index([COLUMNS.location_id, COLUMNS.date])
    data = (pd.concat([data.drop(columns=COLUMNS.ln_age_death_rate), moving_average], axis=1)
            .fillna(method='pad')
            .reset_index())

    # TODO: Remove when we can excavate more of the days stuff.
    data[COLUMNS.days] = (data.groupby(COLUMNS.location_id, as_index=False)
                          .apply(lambda x: pd.Series(range(len(x)), index=x.index, name=COLUMNS.days))
                          .droplevel(0))
    return data


def backcast_all_locations(df: pd.DataFrame, rate_threshold: float) -> pd.DataFrame:
    df = df.copy()
    sort_columns = [COLUMNS.location_id, COLUMNS.country, COLUMNS.location, COLUMNS.date]
    df = df.sort_values(sort_columns).reset_index(drop=True)
    df = drop_lagged_deaths_by_location(df)
    df = add_moving_average_ln_asdr(df, rate_threshold)

    ###############################
    # RE-APPLY SECOND DEATH INDEX #
    ###############################
    # TODO: Important things get their own functions.
    # after we expand out days in the moving average bit, need to check we
    # don't do so for days at the beginning with 2 deaths (happens in
    # Other Counties, WA)
    # make sure we still start at last day of two deaths
    df[COLUMNS.last_day_two] = (df
                                .loc[df[COLUMNS.deaths] == 2]
                                .groupby(COLUMNS.location, as_index=False)[COLUMNS.date]
                                .transform('max'))
    df = df.loc[(df[COLUMNS.last_day_two].isnull()) | (df[COLUMNS.date] == df[COLUMNS.last_day_two])]
    df[COLUMNS.two_date] = df.groupby(COLUMNS.location_id, as_index=False).Date.transform('min')

    # just want second death on, and only where total deaths
    df = df.loc[df[COLUMNS.date] >= df[COLUMNS.two_date]]
    df[COLUMNS.days] = df[COLUMNS.date] - df[COLUMNS.two_date]
    df[COLUMNS.days] = df[COLUMNS.days].apply(lambda x: x.days)
    groupby_cols = [COLUMNS.location_id, COLUMNS.location, COLUMNS.country, COLUMNS.days]
    df = df.sort_values(groupby_cols).reset_index(drop=True)
    ###################################

    # get delta
    obs_diff = df[COLUMNS.obs_ln_age_death_rate].values[1:] - df[COLUMNS.obs_ln_age_death_rate].values[:-1]
    diff = df[COLUMNS.ln_age_death_rate].values[1:] - df[COLUMNS.ln_age_death_rate].values[:-1]
    df[COLUMNS.delta_ln_asdr] = np.nan
    df[COLUMNS.delta_ln_asdr][1:] = diff
    df[COLUMNS.observed_delta_ln_asdr] = np.nan
    df[COLUMNS.observed_delta_ln_asdr][1:] = obs_diff

    groupby_cols = [COLUMNS.location_id, COLUMNS.country, COLUMNS.location]

    df[COLUMNS.first_point] = df.groupby(groupby_cols, as_index=False)[COLUMNS.days].transform('min')
    df.loc[df[COLUMNS.days] == df[COLUMNS.first_point], COLUMNS.delta_ln_asdr] = np.nan
    df.loc[df[COLUMNS.days] == df[COLUMNS.first_point], COLUMNS.observed_delta_ln_asdr] = np.nan

    # project backwards using lagged ln(asdr)
    delta_df = df.copy()
    delta_df = delta_df.loc[(delta_df[COLUMNS.days] > 0) & (delta_df[COLUMNS.days] <= 5)]
    delta_df = delta_df.groupby(groupby_cols, as_index=False)[COLUMNS.delta_ln_asdr].mean()
    not_nursing_home = ~delta_df[COLUMNS.location].isin([LOCATIONS.life_care.name])
    delta_df = delta_df.loc[(delta_df[COLUMNS.delta_ln_asdr] > 1e-4) & not_nursing_home]

    bc_location_ids = delta_df[COLUMNS.location_id].to_list()
    bc_df = pd.concat([
        backcast_log_age_standardized_death_ratio(
            df.loc[df[COLUMNS.location_id] == bc_location_id],
            bc_location_id,
            delta_df.loc[delta_df[COLUMNS.location_id] == bc_location_id, COLUMNS.delta_ln_asdr].item(),
            rate_threshold,
        )
        for bc_location_id in bc_location_ids
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


def get_asdr(true_rate, implied_rate, age_pattern_df: pd.DataFrame):
    scaled_rate = age_pattern_df['death_rate'] * (true_rate / implied_rate)
    return (scaled_rate * age_pattern_df['age_group_weight_value']).sum()


def compute_backcast_log_age_specific_death_rates(df: pd.DataFrame, age_pop_df: pd.DataFrame,
                                                  age_death_df: pd.DataFrame, standardize_location_id: int,
                                                  subnat: bool, rate_threshold: int) -> pd.DataFrame:
    df = process_death_df(df, subnat)

    # get implied death rate based on "standard" population (using average
    # of all possible locations atm)
    standard_age_death_df = get_standard_age_death_df(age_death_df)
    location_to_standardize = age_pop_df[COLUMNS.location_id] == standardize_location_id
    age_pattern_df = age_pop_df.loc[location_to_standardize].merge(standard_age_death_df)

    implied_df = standard_age_death_df.merge(age_pop_df)
    implied_df[COLUMNS.implied_death_rate] = implied_df[COLUMNS.death_rate_bad] * implied_df[COLUMNS.age_group_weight]
    implied_df = implied_df.groupby(COLUMNS.location_id, as_index=False)[COLUMNS.implied_death_rate].sum()
    df = df.merge(implied_df)

    # age-standardize
    df[COLUMNS.age_standardized_death_rate] = df.apply(
        lambda x: get_asdr(
            x[COLUMNS.death_rate],
            x[COLUMNS.implied_death_rate],
            age_pattern_df),
        axis=1)
    df[COLUMNS.ln_age_death_rate] = np.log(df[COLUMNS.age_standardized_death_rate])

    # keep above our threshold death rate, start counting days from there
    df = df.loc[df[COLUMNS.ln_age_death_rate] >= rate_threshold]

    df[COLUMNS.day1] = df.groupby([COLUMNS.country, COLUMNS.location], as_index=False)[COLUMNS.date].transform('min')
    df[COLUMNS.days] = df[COLUMNS.date] - df[COLUMNS.day1]
    df[COLUMNS.days] = df[COLUMNS.days].apply(lambda x: x.days)
    del df[COLUMNS.day1]

    # for Hubei, move it a few days out
    # TODO: document this better.
    if 'Hubei' in df[COLUMNS.location].to_list():
        df.loc[df[COLUMNS.location] == 'Hubei', COLUMNS.days] += 3
        print('Moving Hubei out a few days')

    # interpolate back to threshold
    df = backcast_all_locations(df, rate_threshold)
    return df.reset_index(drop=True)
