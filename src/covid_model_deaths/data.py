from datetime import timedelta
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


sns.set_style('whitegrid')


class _Measures(NamedTuple):
    age_death: str = 'age_death'
    age_pop: str = 'age_pop'
    confirmed: str = 'confirmed'
    deaths: str = 'deaths'
    full_data: str = 'full_data'
    us_pops: str = 'us_pops'


MEASURES = _Measures()

# FIXME: Lots of chained indexing which is error prone and makes pandas
#  mad.


def get_input_data(measure: str, data_version: str = 'best') -> pd.DataFrame:
    """Loads in data for a particular measure and data version.

    Parameters
    ----------
    measure
        The measure to load in.  Should be one of :data:`MEASURES`.
    data_version
        The model input data version to use.  One of 'latest' or 'best'.
        Defaults to 'best', which is the last production ready data set.

    Returns
    -------
        The requested data set as a :class:`pd.DataFrame`.

    """
    if measure not in MEASURES:
        raise ValueError(f'Invalid measure {measure} - valid measures are {", ".join(MEASURES)}')

    df = pd.read_csv(f'/ihme/covid-19/model-inputs/{data_version}/{measure}.csv')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    if 'location_id' in df.columns:
        df['location_id'] = df['location_id'].astype(int)

    return df


def plot_crude_rates(df: pd.DataFrame, level: str = None):
    """Plots crude (population level) rates."""
    # TODO: Generalize and switch to location ids.
    # TODO: Move to plotting module, auto-generate plots in a standard location
    #  during model runs.
    # subset if needed
    if level == 'subnat':
        df = df.loc[df['Location'] != df['Country/Region']].reset_index(drop=True)
    elif level == 'admin0':
        df = df.loc[df['Location'] == df['Country/Region']].reset_index(drop=True)
    elif level is not None:
        raise ValueError('Invalid level specified in plotting call.')

    # do the plotting
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    df = df.sort_values(['Country/Region', 'Location', 'Date'])
    for location, country in df[['Location', 'Country/Region']].drop_duplicates().values:
        # TODO: Make a rule or abstract to a country-style mapping.
        if country == 'China':
            style = '--'
        elif country == 'United States of America':
            style = ':'
        elif country == 'Italy':
            style = '-.'
        elif country == 'Spain':
            style = (0, (1, 10))
        elif country == 'Germany':
            style = (0, (5, 10))
        else:
            style = '-'
        plt.plot(df.loc[df['Location'] == location, 'Days'],
                 np.log(df.loc[df['Location'] == location, 'Death rate']),
                 label=location, linestyle=style)
    plt.xlabel('Days')
    plt.xlim(0, df['Days'].max())
    plt.ylabel('ln(death rate)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    plt.show()


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
        df = pd.concat(
            [moving_average(loc_df, 'ln(age-standardized death rate)', self.rate_threshold) for loc_df in loc_df_list]
        ).reset_index(drop=True)

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


def moving_average(df: pd.DataFrame, rate_var: str, rate_threshold: int = None, reset_days: bool = False) -> pd.DataFrame:
    if reset_days:
        df['Days'] -= df['Days'].min()
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
    def moving_3day_avg(day, df):
        # determine difference
        days = np.array([day-1, day, day+1])
        days = days[days >= 0]
        days = days[days <= df['Days'].max()]
        avg = df.loc[df['Days'].isin(days), rate_var].mean()

        return avg

    # get diffs
    avgs = [moving_3day_avg(i, df) for i in df['Days']]
    df[f'Observed {rate_var}'] = df[rate_var]
    df[rate_var] = avgs

    # replace last point w/ daily value over 3->2 and 2->1 and the first
    # with 1->2, 2->3; use observed if 3 data points or less
    if len(df) > 3:
        last_step = np.mean(np.array(avgs[-3:-1]) - np.array(avgs[-4:-2]))
        df[rate_var][len(df)-1] = (df[rate_var][len(df)-2]
                                                            + last_step)
        first_step = np.mean(np.array(avgs[2:4]) - np.array(avgs[1:3]))
        df[rate_var][0] = df[rate_var][1] - first_step
        if rate_threshold is not None and df[rate_var][0] < rate_threshold:
            df[rate_var][0] = rate_threshold
    else:
        df[rate_var] = df[f'Observed {rate_var}']

    return df


class LeadingIndicator:
    def __init__(self, full_df: pd.DataFrame, data_version: str = 'best'):
        # expect passed in file to be `full_data.csv`
        self.data_version = data_version
        self.full_df = self._clean_up_dataset(full_df)
        
    def _tests_per_capita(self, pop_df: pd.DataFrame) -> pd.DataFrame:
        # load testing data from snapshot
        us_df = pd.read_csv(f'/ihme/covid-19/snapshot-data/{self.data_version}/covid_onedrive/Testing/us_states_tests.csv')
        us_df['Date'] = pd.to_datetime(us_df['date'], format='%d.%m.%Y')
        us_df = us_df.rename(index=str, columns={'totaltestresults':'Tests'})
        g_df = pd.read_csv(f'/ihme/covid-19/snapshot-data/{self.data_version}/covid_onedrive/Testing/global_admin0_tests.csv')
        g_df['Date'] = pd.to_datetime(g_df['date'], format='%d.%m.%Y')
        g_df = g_df.rename(index=str, columns={'total_tests':'Tests'})
        df = us_df[['location_id', 'Date', 'Tests']].append(g_df[['location_id', 'date', 'Tests']])
        #df = pd.read_csv('/home/j/temp/kcausey/covid19/test_prop/data_smooth_4_27_global.csv')
        #df = df.rename(index=str, columns={'daily_total':'Tests'})
        #df['Date'] = pd.to_datetime(df['date'])
        
        # format and get testing rate
        df = df.loc[(~df['location_id'].isnull()) & (~df['Tests'].isnull())]
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        #df['Tests'] = df.groupby('location_id', as_index=False)['Tests'].cumsum()
        df = df.merge(pop_df)
        df['Testing rate'] = df['Tests'] / df['population']
        
        # smooth
        df['ln(testing rate)'] = np.log(df['Testing rate'])
        df.loc[df['Tests'] == 0, 'ln(testing rate)'] = np.log(0.1 / df['population'])
        df['day0'] = df.groupby('location_id', as_index=False)['Date'].transform(min)
        df['Days'] = df.apply(lambda x: (x['Date'] - x['day0']).days, axis=1)
        df = self._smooth_data(df, 'ln(testing rate)')
        df['location_id'] = df['location_id'].astype(int)
        df['Testing rate'] = np.exp(df['ln(testing rate)'])
        
        return df[['location_id', 'Date', 'Tests', 'Testing rate']]
    
    def _account_for_positivity(self, t1: float, t2: float, 
                                c1: float, c2: float,
                                logit_pos_int: float = -1.67,
                                logit_pos_logit_test: float = -0.643) -> float:
        '''
        #t1 = 0.0009422044413620219
        t1 = 500 / 1e6
        #t2 = 0.0005275541285749835
        t2 = 800 / 1e6
        #c1 = 3.651402444897634e-05
        c1 = 100 / 1e6
        #c2 = 4.6986292647525517e-05
        c2 = 180 / 1e6
        logit_pos_int = -8.05
        logit_pos_logit_test = -0.78
        '''
        # if c2 <= c1 or t2 <= t1:
        #     # should we do something about reduced testing?
        #     cases = c2
        # else:
        logit = lambda x: np.log(x / (1 - x))
        expit = lambda x: 1/(1 + np.exp(-x))
        
        p1 = expit(logit_pos_int + logit_pos_logit_test * logit(t1))
        p2 = expit(logit_pos_int + logit_pos_logit_test * logit(t2))
        increase_from_testing = (t2*p2 - t1*p1) / (t1*p1)
        #increase_from_testing = ((t2-t1) / t1) * (1 - positivity_effect)
        
        excess_reporting = ((c2-c1) / c1) - increase_from_testing
        
        #if excess_reporting < 0:
        #    excess_reporting = 0
        cases = c1 * (1 + excess_reporting)
            
        if cases < 0:
            cases = 0
        
        return cases
    
    def _control_for_testing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df['Daily testing rate'] = np.nan
        df['Daily testing rate'][1:] = df['Testing rate'].values[1:] - df['Testing rate'].values[:-1]
        df['Daily case rate'] = np.nan
        df['Daily case rate'][1:] = df['Confirmed case rate'].values[1:] - df['Confirmed case rate'].values[:-1]
        
        # keep last 8 days
        future_df = df.loc[df['Date'] >= df['Date'].max() - timedelta(days=8)]
        
        # use case data if it is less than 3 days behind
        if future_df['Testing rate'].isnull().sum() < 3:
            future_df = future_df.sort_values('Date').reset_index(drop=True)
            future_df = future_df.loc[~future_df['Testing rate'].isnull()]
            start_cases = future_df['Daily case rate'][0]
            start_tests = future_df['Daily testing rate'][0]
            #testing = stats.linregress(future_df.index.values, np.log(future_df['Testing rate'].values))
            #future_df['Testing estimate'] = np.exp(testing.intercept + testing.slope * future_df.index.values)

            future_df['Adjusted daily case rate'] = future_df.apply(
                lambda x: self._account_for_positivity(start_tests, x['Daily testing rate'],
                                                       start_cases, x['Daily case rate']),
                axis=1
            )
            future_df['Adjusted daily case rate'][0] = future_df['Confirmed case rate'][0]
            future_df['Adjusted case rate'] = future_df['Adjusted daily case rate'].cumsum()
            df = pd.concat([df.loc[df['Date'] < df['Date'].max() - timedelta(days=8)], 
                            future_df]).reset_index(drop=True)
        
        return df
        
    def _clean_up_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # hosptalization rate not in data
        df['Hospitalization rate'] = df['Hospitalizations'] / df['population']
        
        # id
        df['location_id'] = df['location_id'].astype(int)
        
        # get days
        df['day0'] = df.groupby('location_id', as_index=False)['Date'].transform(min)
        df['Days'] = df.apply(lambda x: (x['Date'] - x['day0']).days, axis=1)
        
        # get ln
        df.loc[df['Confirmed case rate'] == 0, 'Confirmed case rate'] = 0.1 / df['population']
        df.loc[df['Hospitalization rate'] == 0, 'Hospitalizations rate'] = 0.1 / df['population']
        df.loc[df['Death rate'] == 0, 'Death rate'] = 0.1 / df['population']
        df['ln(confirmed case rate)'] = np.log(df['Confirmed case rate'])
        df['ln(hospitalization rate)'] = np.log(df['Hospitalization rate'])
        df['ln(death rate)'] = np.log(df['Death rate'])
        df = df[['location_id', 'Date', 'Days', 'population', 
                 'Confirmed', 'Confirmed case rate', 'ln(confirmed case rate)', 
                 'Hospitalizations', 'Hospitalization rate', 'ln(hospitalization rate)', 
                 'Deaths', 'Death rate', 'ln(death rate)']].sort_values(['location_id', 'Date']).reset_index(drop=True)
        
        return df
    
    def _smooth_data(self, df: pd.DataFrame, smooth_var: str, reset_days: bool = False) -> pd.DataFrame:
        df = df.copy()
        loc_dfs = [df.loc[df['location_id'] == l].reset_index(drop=True) for l in df.location_id.unique()]
        loc_dfs = [moving_average(loc_df, smooth_var, reset_days=reset_days) for loc_df in loc_dfs]
        if loc_dfs:
            df = pd.concat(loc_dfs)
        
        return df
    
    def _average_over_last_days(self, df: pd.DataFrame, avg_var: str, mean_window: int = 3) -> pd.DataFrame:
        df['latest date'] = df.groupby('location_id', as_index=False)['Date'].transform(max)
        df['group days'] = df['group date'].apply(lambda x: x - timedelta(days=mean_window-1))
        df = df.loc[df['Date'] >= df['last days']]
        df = df.groupby('location_id', as_index=False)[avg_var].mean()
        
        return df
    
    def _get_death_to_prior_indicator(self) -> pd.DataFrame:
        # smooth cases, prepare to match with deaths 8 days later
        case_df = self._smooth_data(self.full_df, 'ln(confirmed case rate)')
        case_df['Confirmed case rate'] = np.exp(case_df['ln(confirmed case rate)'])
        case_df['Date'] = case_df['Date'].apply(lambda x: x + timedelta(days=8))
        full_case_df = case_df[['location_id', 'Date', 'Confirmed case rate']].copy()
        #case_df = case_df.loc[case_df['Confirmed'] > 0].reset_index(drop=True)

        # do the same thing with hospitalizations (not present for all locs, so subset)
        hosp_df = self._smooth_data(self.full_df.loc[~self.full_df['Hospitalizations'].isnull()], 
                                    'ln(hospitalization rate)',
                                    reset_days=True)
        hosp_df['Hospitalization rate'] = np.exp(hosp_df['ln(hospitalization rate)'])
        hosp_df['Date'] = hosp_df['Date'].apply(lambda x: x + timedelta(days=8))
        full_hosp_df = hosp_df[['location_id', 'Date', 'Hospitalization rate']].copy()
        #hosp_df = hosp_df.loc[hosp_df['Hospitalizations'] > 0].reset_index(drop=True)

        # smooth deaths
        death_df = self._smooth_data(self.full_df, 'ln(death rate)')
        death_df['Death rate'] = np.exp(death_df['ln(death rate)'])
        full_death_df = death_df[['location_id', 'Date', 'Deaths', 'Death rate']].copy()
        #death_df = death_df.loc[death_df['Deaths'] > 0].reset_index(drop=True)
        
        # calc ratios by day
        dcr_df = death_df[['location_id', 'Date', 'Death rate']].merge(
            case_df[['location_id', 'Date', 'Confirmed case rate']]
        )
        dhr_df = death_df[['location_id', 'Date', 'Death rate']].merge(
            hosp_df[['location_id', 'Date', 'Hospitalization rate']]
        )
        dcr_df['dcr lag8'] = dcr_df['Death rate'] / dcr_df['Confirmed case rate']
        dhr_df['dhr lag8'] = dhr_df['Death rate'] / dhr_df['Hospitalization rate']
        
        # average ratios of last 3 days
        dcr_df = self._average_over_last_days(dcr_df, 'dcr lag8', 3)
        if not dhr_df.empty:
            dhr_df = self._average_over_last_days(dhr_df, 'dhr lag8', 3)
        else:
            dhr_df = dhr_df[['location_id', 'dhr lag8']]
        
        return full_case_df, full_hosp_df, full_death_df, dcr_df, dhr_df
    
    def _stream_out_deaths(self, df: pd.DataFrame, death_df: pd.DataFrame) -> pd.DataFrame:
        # convert to daily, fix to last day of observed deaths
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        daily = df['Death rate'].values[1:] - df['Death rate'].values[:-1]
        df['Daily death rate'] = np.nan
        df['Daily death rate'][1:] = daily
        del df['Death rate']
        df = df.merge(death_df[['location_id', 'last date', 'Death rate']])
        df = df.loc[df['Date'] > df['last date']]  # will exclude days where diff is one locations last day and anothers first
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        df['Death rate'] = df.groupby('location_id', as_index=False)['Daily death rate'].cumsum().values + \
                           df[['Death rate']].values
        
        return df
    
    def _combine_data(self, df: pd.DataFrame, rate_var: str, ratio_var: str, 
                      ratio_df: pd.DataFrame, na_fill: float) -> pd.DataFrame:
        df = df.merge(ratio_df, how='left')
        df['location_id'] = df['location_id'].astype(int)
        df[ratio_var] = df[ratio_var].fillna(na_fill)
        df['Death rate'] = df[rate_var] * df[ratio_var]
        del df[rate_var]
        del df[ratio_var]
        
        return df
    
    def _combine_sources(self, df):
        # only use hospital if it is less than 3 days behind
        if df['from_hospital'].isnull().sum() < 3:
            df = df.loc[~df['from_hospital'].isnull()]
            df['Death rate'] = df[['from_cases', 'from_hospital']].mean(axis=1)
            df['source'] = 'cases+hospital'
        else:
            df['Death rate'] = df['from_cases']
            df['source'] = 'cases'
            
        return df
    
    def produce_deaths(self) -> pd.DataFrame:
        # get ratio
        case_df, hosp_df, death_df, dcr_df, dhr_df = self._get_death_to_prior_indicator()
        
        # get last day of observed deaths and smoothed death rate
        death_df['last date'] = death_df.groupby('location_id', as_index=False)['Date'].transform(max)
        death_df = death_df.loc[death_df['Date'] == death_df['last date']]
        death_df = death_df[['location_id', 'last date', 'Deaths', 'Death rate']]
        
        # set limits on death-to-case ratio based on number of deaths...
        #   - if >= 10, use 2.5th/97.5th of ratios in places with more than
        #     30 deaths (0.02, 0.2)
        #   - if < 10, use 10th/90th of that group (0.03, 0.15)
        dcr_df = dcr_df.merge(death_df[['location_id', 'Deaths']])
        dcr_df.loc[(dcr_df['Deaths'] >= 10) & (dcr_df['dcr lag8'] < 0.02), 'dcr lag8'] = 0.02
        dcr_df.loc[(dcr_df['Deaths'] >= 10) & (dcr_df['dcr lag8'] > 0.2), 'dcr lag8'] = 0.2
        dcr_df.loc[(dcr_df['Deaths'] < 10) & (dcr_df['dcr lag8'] < 0.03), 'dcr lag8'] = 0.03
        dcr_df.loc[(dcr_df['Deaths'] < 10) & (dcr_df['dcr lag8'] > 0.15), 'dcr lag8'] = 0.15
        del dcr_df['Deaths']
        dcr_df['location_id'] = dcr_df['location_id'].astype(int)
        
        # apply ratio to get deaths (use <10 deaths floor as ratio for places without deaths thus far)
        dc_df = self._combine_data(case_df, 'Confirmed case rate', 'dcr lag8', 
                                   dcr_df[['location_id', 'dcr lag8']], 0.03)
        dh_df = self._combine_data(hosp_df, 'Hospitalization rate', 'dhr lag8', 
                                   dhr_df[['location_id', 'dhr lag8']], 0.1)
        
        # start daily deaths from last data point
        death_df['location_id'] = death_df['location_id'].astype(int)
        dc_df = self._stream_out_deaths(dc_df, death_df)
        dc_df = dc_df.rename(index=str, columns={'Death rate': 'from_cases'})
        dh_df = self._stream_out_deaths(dh_df, death_df)
        dh_df = dh_df.rename(index=str, columns={'Death rate': 'from_hospital'})
        df = dc_df[['location_id', 'Date', 'from_cases']].merge(
            dh_df[['location_id', 'Date', 'from_hospital']],
            how='outer'
        )
        
        # combine locations
        df = pd.concat(
            [self._combine_sources(df.loc[df['location_id'] == l]) for l in df['location_id'].unique()]
        ).reset_index(drop=True)
        df['ln(age-standardized death rate)'] = np.log(df['Death rate'])
        
        return dcr_df, dhr_df, df[['location_id', 'Date', 'ln(age-standardized death rate)', 
                                   'from_cases', 'from_hospital', 'source']]
        