import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS, LOCATIONS
from covid_model_deaths.preprocessing import expanding_moving_average_by_location, backcast_all_locations


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

    sort_columns = [COLUMNS.location_id, COLUMNS.country, COLUMNS.location, COLUMNS.date]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    df = drop_lagged_deaths_by_location(df)
    df = add_moving_average_ln_asdr(df, rate_threshold)
    df = add_days_since_last_day_of_two_deaths(df)

    # interpolate back to threshold
    df = backcast_all_locations(df, rate_threshold)
    return df.reset_index(drop=True)


####################
# Helper functions #
####################
# TODO:
#  - Excavate library code.
#  - Vectorize where possible.
#  - Leave behind processing specific to produce the computation
#  - Move processing-specific code to application package
#  - Wrap in cli.

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


def get_asdr(true_rate, implied_rate, age_pattern_df: pd.DataFrame):
    scaled_rate = age_pattern_df['death_rate'] * (true_rate / implied_rate)
    return (scaled_rate * age_pattern_df['age_group_weight_value']).sum()


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


def add_moving_average_ln_asdr(data: pd.DataFrame, rate_threshold: float, n_smooths: int = 3) -> pd.DataFrame:
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
    required_columns = [COLUMNS.location_id, COLUMNS.date, COLUMNS.days, COLUMNS.ln_age_death_rate]
    assert set(required_columns).issubset(data.columns)
    data[COLUMNS.obs_ln_age_death_rate] = data[COLUMNS.ln_age_death_rate]
    # smooth three times
    moving_average = data.copy()
    for i in range(n_smooths):
        moving_average = expanding_moving_average_by_location(moving_average, COLUMNS.ln_age_death_rate)
    # noinspection PyTypeChecker
    moving_average[moving_average < rate_threshold] = rate_threshold
    data = data.set_index([COLUMNS.location_id, COLUMNS.date])
    data = (pd.concat([data.drop(columns=COLUMNS.ln_age_death_rate), moving_average], axis=1)
            .fillna(method='pad')
            .reset_index())

    return data


def add_days_since_last_day_of_two_deaths(data: pd.DataFrame) -> pd.DataFrame:
    """Compute days since the last day with two deaths by location."""
    # after we expand out days in the moving average bit, need to check we
    # don't do so for days at the beginning with 2 deaths (happens in
    # Other Counties, WA)
    # make sure we still start at last day of two deaths
    required_columns = [COLUMNS.location_id, COLUMNS.date, COLUMNS.deaths]
    assert set(required_columns).issubset(data.columns)
    data['last_day_two'] = (data
                            .loc[data[COLUMNS.deaths] == 2]
                            .groupby(COLUMNS.location_id, as_index=False)[COLUMNS.date]
                            .transform('max'))
    data = data.loc[(data['last_day_two'].isnull()) | (data[COLUMNS.date] == data['last_day_two'])]
    data['two_date'] = data.groupby(COLUMNS.location_id, as_index=False)[COLUMNS.date].transform('min')

    # just want second death on, and only where total deaths
    data = data.loc[data[COLUMNS.date] >= data['two_date']]
    data[COLUMNS.days] = data[COLUMNS.date] - data['two_date']
    data[COLUMNS.days] = data[COLUMNS.days].dt.days
    data = data.sort_values([COLUMNS.location_id, COLUMNS.date]).reset_index(drop=True)
    # FIXME: I'm like 90% sure these columns aren't used anywhere else.
    #  But they get written to outputs.
    # del data['last_day_two']
    # del data['two_date']
    return data
