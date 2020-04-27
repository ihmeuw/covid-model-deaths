from datetime import timedelta

import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS, LOCATIONS


def process_death_df(death_df: pd.DataFrame, subnat: bool) -> pd.DataFrame:
    """Subset and filter the death data."""
    death_df = death_df.sort_values([COLUMNS.country, COLUMNS.location, COLUMNS.date]).reset_index(drop=True)

    if subnat:
        # FIXME: Faulty logic.  Use location hierarchy
        location_matches_country = death_df[COLUMNS.location] == death_df[COLUMNS.country]
        death_df = death_df.loc[~location_matches_country].reset_index(drop=True)

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


def moving_3day_avg(day, data):
    # determine difference
    days = np.array([day-1, day, day+1])
    days = days[days >= 0]
    days = days[days <= data[COLUMNS.days].max()]
    avg = data.loc[data[COLUMNS.days].isin(days), COLUMNS.ln_age_death_rate].mean()
    return avg


def moving_average_log_age_standardized_death_ratio(df: pd.DataFrame, rate_threshold: float) -> pd.DataFrame:
    if df[COLUMNS.location_id].unique().size != 1:
        raise ValueError('Multiple locations in dataset.')
    if df[COLUMNS.days].min() != 0:
        raise ValueError('Not starting at 0')

    full_day_range = pd.DataFrame({COLUMNS.days: np.arange(df[COLUMNS.days].min(), df[COLUMNS.days].max()+1)})
    df = df.merge(full_day_range, how='outer')
    df = df.sort_values(COLUMNS.days).reset_index(drop=True)

    no_date = df[COLUMNS.date].isnull()
    df.loc[no_date, COLUMNS.date] = (df
                                     .loc[no_date, COLUMNS.days]
                                     .apply(lambda x: df[COLUMNS.date].min() + timedelta(days=x)))
    # TODO: Document.
    df = df.fillna(method='pad')

    # get diffs
    moving_average = [moving_3day_avg(i, df) for i in df[COLUMNS.days]]
    df[COLUMNS.obs_ln_age_death_rate] = df[COLUMNS.ln_age_death_rate]
    df[COLUMNS.ln_age_death_rate] = moving_average

    # replace last point w/ daily value over 3->2 and 2->1 and the first
    # with 1->2, 2->3; use observed if 3 data points or less
    if len(df) > 3:
        last_step = np.mean(np.array(moving_average[-3:-1]) - np.array(moving_average[-4:-2]))
        df[COLUMNS.ln_age_death_rate][len(df)-1] = (df[COLUMNS.ln_age_death_rate][len(df)-2] + last_step)

        first_step = np.mean(np.array(moving_average[2:4]) - np.array(moving_average[1:3]))
        df[COLUMNS.ln_age_death_rate][0] = df[COLUMNS.ln_age_death_rate][1] - first_step
        if df[COLUMNS.ln_age_death_rate][0] < rate_threshold:
            df[COLUMNS.ln_age_death_rate][0] = rate_threshold
    else:
        df[COLUMNS.ln_age_death_rate] = df[COLUMNS.obs_ln_age_death_rate]

    return df


class DeathModelData:
    """Wrapper class that does mortality rate back-casting."""

    # TODO: Pull out data processing separately from modeling.
    # TODO: rate threshold global.
    def __init__(self, df: pd.DataFrame, age_pop_df: pd.DataFrame, age_death_df: pd.DataFrame,
                 standardize_location_id: int, subnat: bool = False,
                 rate_threshold: int = -15):
        """
        Parameters
        ----------
        df
        age_pop_df
        age_death_df
        standardize_location_id
        subnat
        rate_threshold

        """
        # set rate
        self.rate_threshold = rate_threshold
        df = process_death_df(df, subnat)

        # get implied death rate based on "standard" population (using average
        # of all possible locations atm)
        standard_age_death_df = get_standard_age_death_df(age_death_df)
        location_to_standardize = age_pop_df[COLUMNS.location_id] == standardize_location_id
        age_pattern_df = age_pop_df.loc[location_to_standardize].merge(standard_age_death_df)

        implied_df = standard_age_death_df.merge(age_pop_df)
        implied_df['Implied death rate'] = implied_df['death_rate'] * implied_df['age_group_weight_value']
        implied_df = implied_df.groupby('location_id', as_index=False)['Implied death rate'].sum()
        df = df.merge(implied_df)

        # age-standardize
        df['Age-standardized death rate'] = df.apply(
            lambda x: self.get_asdr(
                x['Death rate'],
                x['Implied death rate'],
                age_pattern_df),
            axis=1)
        df['ln(age-standardized death rate)'] = np.log(df['Age-standardized death rate'])

        # keep above our threshold death rate, start counting days from there
        df = df.loc[df['ln(age-standardized death rate)'] >= rate_threshold]
        df['Day1'] = df.groupby(['Country/Region', 'Location'], as_index=False)['Date'].transform('min')
        df['Days'] = df['Date'] - df['Day1']
        df['Days'] = df['Days'].apply(lambda x: x.days)
        del df['Day1']

        # for Hubei, move it a few days out
        # TODO: document this better.
        if 'Hubei' in df['Location'].to_list():
            df.loc[df['Location'] == 'Hubei', 'Days'] += 3
            print('Moving Hubei out a few days')

        # interpolate back to threshold
        df = self._backcast_all_locations(df)

        self.dep_var = 'ln(age-standardized death rate)'
        self.df = df.reset_index(drop=True)

    def _backcast_all_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        # sort dataset
        df = df.copy()
        df = df.sort_values(['location_id', 'Country/Region', 'Location', 'Date']).reset_index(drop=True)

        # get delta, drop last day if it does not contain new deaths (assume lag)
        diff = df['ln(age-standardized death rate)'].values[1:] - df['ln(age-standardized death rate)'].values[:-1]
        df['Delta ln(asdr)'] = np.nan
        df['Delta ln(asdr)'][1:] = diff
        df['first_point'] = df.groupby(['location_id', 'Country/Region', 'Location'], as_index=False).Days.transform(
            'min')
        df.loc[df['Days'] == df['first_point'], 'Delta ln(asdr)'] = np.nan
        df['last_point'] = df.groupby(['location_id', 'Country/Region', 'Location'], as_index=False).Days.transform(
            'max')
        df = df.loc[~((df['Days'] == df['last_point']) & (df['Delta ln(asdr)'] == 0))]

        # clean up (will add delta back after expanding)
        del df['first_point']
        del df['last_point']
        del df['Delta ln(asdr)']

        # fill in missing days and smooth
        loc_df_list = [df.loc[df['location_id'] == loc_id] for loc_id in df['location_id'].unique()]
        df = pd.concat([moving_average_log_age_standardized_death_ratio(loc_df, self.rate_threshold)
                        for loc_df in loc_df_list]).reset_index(drop=True)

        ###############################
        # RE-APPLY SECOND DEATH INDEX #
        ###############################
        # TODO: Important things get their own functions.
        # after we expand out days in the moving average bit, need to check we
        # don't do so for days at the beginning with 2 deaths (happens in
        # Other Counties, WA)
        # make sure we still start at last day of two deaths
        df['last_day_two'] = df.loc[df['Deaths'] == 2].groupby('location_id', as_index=False).Date.transform('max')
        df = df.loc[(df['last_day_two'].isnull()) | (df['Date'] == df['last_day_two'])]
        df['two_date'] = df.groupby('location_id', as_index=False).Date.transform('min')

        # just want second death on, and only where total deaths
        df = df.loc[df['Date'] >= df['two_date']]
        df['Days'] = df['Date'] - df['two_date']
        df['Days'] = df['Days'].apply(lambda x: x.days)
        df = df.sort_values(['location_id', 'Location', 'Country/Region', 'Days']).reset_index(drop=True)
        ###################################

        # get delta
        obs_diff = (df['Observed ln(age-standardized death rate)'].values[1:]
                    - df['Observed ln(age-standardized death rate)'].values[:-1])
        diff = df['ln(age-standardized death rate)'].values[1:] - df['ln(age-standardized death rate)'].values[:-1]
        df['Delta ln(asdr)'] = np.nan
        df['Delta ln(asdr)'][1:] = diff
        df['Observed delta ln(asdr)'] = np.nan
        df['Observed delta ln(asdr)'][1:] = obs_diff
        df['first_point'] = (df
                             .groupby(['location_id', 'Country/Region', 'Location'], as_index=False)
                             .Days.transform('min'))
        df.loc[df['Days'] == df['first_point'], 'Delta ln(asdr)'] = np.nan
        df.loc[df['Days'] == df['first_point'], 'Observed delta ln(asdr)'] = np.nan

        # project backwards using lagged ln(asdr)
        delta_df = df.copy()
        delta_df = delta_df.loc[(delta_df['Days'] > 0) & (delta_df['Days'] <= 5)]
        delta_df = (delta_df
                    .groupby(['location_id', 'Country/Region', 'Location'], as_index=False)['Delta ln(asdr)']
                    .mean())
        delta_df = delta_df.loc[(delta_df['Delta ln(asdr)'] > 1e-4) &
                                (~delta_df['Location'].isin(['Life Care Center, Kirkland, WA']))]
        bc_location_ids = delta_df['location_id'].to_list()
        bc_df = pd.concat([
            backcast_log_age_standardized_death_ratio(
                df.loc[df['location_id'] == bc_location_id],
                bc_location_id,
                delta_df.loc[delta_df['location_id'] == bc_location_id, 'Delta ln(asdr)'].item(),
                self.rate_threshold,
            )
            for bc_location_id in bc_location_ids
        ])
        df = df.append(bc_df)
        df = df.sort_values(['location_id', 'Days']).reset_index(drop=True)
        # TODO: Document this assumption about back-filling.
        df[['Location', 'Country/Region', 'population']] = (df[['Location', 'Country/Region', 'population']]
                                                            .fillna(method='backfill'))
        df['location_id'] = df['location_id'].astype(int)
        df['first_point'] = df.groupby(['Country/Region', 'Location'], as_index=False).Days.transform('min')
        df.loc[df['first_point'] < 0, 'Days'] = df['Days'] - df['first_point']
        del df['first_point']

        return df

    @staticmethod
    def get_asdr(true_rate, implied_rate, age_pattern_df: pd.DataFrame):
        scaled_rate = age_pattern_df['death_rate'] * (true_rate / implied_rate)
        return (scaled_rate * age_pattern_df['age_group_weight_value']).sum()


