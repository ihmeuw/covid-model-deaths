from datetime import timedelta

import numpy as np
import pandas as pd

# FIXME: Lots of chained indexing which is error prone and makes pandas
#  mad.


class DeathModelData:
    """Wrapper class that does mortality rate backcasting."""

    # TODO: Pull out data processing separately from modeling.
    # TODO: rate threshold global.
    def __init__(self, df: pd.DataFrame, age_pop_df: pd.DataFrame, age_death_df: pd.DataFrame,
                 standardize_location_id: int, model_type: str, subnat: bool = False,
                 rate_threshold: int = -15):
        """
        Parameters
        ----------
        df
        age_pop_df
        age_death_df
        standardize_location_id
        model_type
        subnat
        rate_threshold

        """
        # set rate
        self.rate_threshold = rate_threshold

        # get model dataset
        # TODO: remove the arg?
        if model_type != 'threshold':
            raise ValueError('Must set model type to be threshold.')
        df = df.sort_values(['Country/Region', 'Location', 'Date']).reset_index(drop=True)

        # restrict subnat if needed
        if subnat:
            # this logic should be sound...?
            df = df.loc[df['Location'] != df['Country/Region']].reset_index(drop=True)

        df = df.loc[df['Location'] != 'Outside Wuhan City, Hubei'].reset_index(drop=True)

        df = df.loc[df['Location'] != 'Outside Hubei'].reset_index(drop=True)

        # make sure we don't have naming problem
        # TODO: Check preconditions on data sets well before this.  Use
        #  proper errors.
        assert (len(df[['location_id']].drop_duplicates())
                == len(df[['Country/Region', 'Location']].drop_duplicates())
                == len(df[['location_id', 'Country/Region', 'Location']].drop_duplicates())), (
            'Location, location_id, Country/Region + Location, and location_id + Country/Region + Location '
            'are not all acceptible keys. I assume this is true, check why not.'
        )

        # get implied death rate based on "standard" population (using average
        # of all possible locations atm)
        implied_df = (age_death_df
                      .loc[age_death_df.location_id == 1, ['age_group', 'death_rate']]
                      .reset_index(drop=True))
        # attach to age pattern for scaling in asdr function
        self.age_pattern_df = age_pop_df.loc[age_pop_df.location_id == standardize_location_id].merge(implied_df)
        implied_df = implied_df.merge(age_pop_df)
        implied_df['Implied death rate'] = implied_df['death_rate'] * implied_df['age_group_weight_value']
        implied_df = implied_df.groupby('location_id', as_index=False)['Implied death rate'].sum()
        df = df.merge(implied_df)

        # age-standardize
        df['Age-standardized death rate'] = df.apply(
            lambda x: self.get_asdr(
                x['Death rate'],
                x['Implied death rate'],
                self.age_pattern_df),
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
        df = pd.concat([self._moving_average_lnasdr(loc_df) for loc_df in loc_df_list]).reset_index(drop=True)

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
            self._backcast_ln_asdr(bc_location_id,
                                   df.loc[df['location_id'] == bc_location_id],
                                   delta_df.loc[delta_df['location_id'] == bc_location_id, 'Delta ln(asdr)'].item())
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

    def _backcast_ln_asdr(self, location_id: int, df: pd.DataFrame, bc_step: int) -> pd.DataFrame:
        # get first point
        start_rep = df.sort_values('Days').reset_index(drop=True)['ln(age-standardized death rate)'][0]

        # backcast if above threshold (already dropped below, so other case
        # is == threshold)
        if start_rep > self.rate_threshold:
            # count from threshold on
            bc_rates = np.arange(self.rate_threshold, start_rep, bc_step)
            bc_df = pd.DataFrame({
                'location_id': location_id,
                'ln(age-standardized death rate)': np.flip(bc_rates)
            })

            # remove fractional step from last  (we force the threshold day to
            # be 0, so the partial day ends up getting added onto the first
            # day) no longer add date, since we have partial days
            if df['Days'].min() != 0:
                raise ValueError(f'First day is not 0, as expected... (location_id: {location_id})')
            bc_df['Days'] = -bc_df.index - (start_rep - bc_rates[-1]) / bc_step

            # don't project more than 10 days back, or we will have PROBLEMS
            bc_df = (bc_df
                     .loc[bc_df['Days'] >= -10, ['location_id', 'Days', 'ln(age-standardized death rate)']]
                     .reset_index(drop=True))
        else:
            assert start_rep == self.rate_threshold, 'First value is below threshold, should not be possible.'
            bc_df = pd.DataFrame(columns=['location_id', 'Days', 'ln(age-standardized death rate)'])

        return bc_df

    @staticmethod
    def get_asdr(true_rate, implied_rate, age_pattern_df: pd.DataFrame):
        scaled_rate = age_pattern_df['death_rate'] * (true_rate / implied_rate)

        return (scaled_rate * age_pattern_df['age_group_weight_value']).sum()

    def _moving_average_lnasdr(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.location_id.unique().size != 1:
            raise ValueError('Multiple locations in dataset.')
        if df['Days'].min() != 0:
            raise ValueError('Not starting at 0')
        df = df.merge(pd.DataFrame({'Days': np.arange(df['Days'].min(), df['Days'].max()+1)}), how='outer')
        df = df.sort_values('Days').reset_index(drop=True)
        df.loc[df['Date'].isnull(), 'Date'] = (df.loc[df['Date'].isnull(), 'Days']
                                               .apply(lambda x: df['Date'].min() + timedelta(days=x)))
        # TODO: Document.
        df = df.fillna(method='pad')

        # FIXME: Shadowing variable from outer scope.  Make a separate
        #  function.
        def moving_3day_avg(day, data):
            # determine difference
            days = np.array([day-1, day, day+1])
            days = days[days >= 0]
            days = days[days <= data['Days'].max()]
            avg = data.loc[data['Days'].isin(days), 'ln(age-standardized death rate)'].mean()

            return avg

        # get diffs
        avgs = [moving_3day_avg(i, df) for i in df['Days']]
        df['Observed ln(age-standardized death rate)'] = df['ln(age-standardized death rate)']
        df['ln(age-standardized death rate)'] = avgs

        # replace last point w/ daily value over 3->2 and 2->1 and the first
        # with 1->2, 2->3; use observed if 3 data points or less
        if len(df) > 3:
            last_step = np.mean(np.array(avgs[-3:-1]) - np.array(avgs[-4:-2]))
            df['ln(age-standardized death rate)'][len(df)-1] = (df['ln(age-standardized death rate)'][len(df)-2]
                                                                + last_step)
            first_step = np.mean(np.array(avgs[2:4]) - np.array(avgs[1:3]))
            df['ln(age-standardized death rate)'][0] = df['ln(age-standardized death rate)'][1] - first_step
            if df['ln(age-standardized death rate)'][0] < self.rate_threshold:
                df['ln(age-standardized death rate)'][0] = self.rate_threshold
        else:
            df['ln(age-standardized death rate)'] = df['Observed ln(age-standardized death rate)']

        return df
