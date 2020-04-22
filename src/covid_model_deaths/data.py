from typing import NamedTuple

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_style('whitegrid')

import numpy as np
import pandas as pd


# pd.options.display.max_columns = 9999
# pd.options.display.max_rows = 9999


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
            #print('Only using admin1 and below locations')
            df = df.loc[df['Location'] != df['Country/Region']].reset_index(drop=True)

        #print('Dropping Outside Wuhan City, Hubei')
        df = df.loc[df['Location'] != 'Outside Wuhan City, Hubei'].reset_index(drop=True)

        #print('Dropping Outside Hubei')
        df = df.loc[df['Location'] != 'Outside Hubei'].reset_index(drop=True)

        # make sure we don't have naming problem
        # TODO: Check preconditions on data sets well before this.  Use
        #  proper errors.
        assert (len(df[['Location']].drop_duplicates())
                == len(df[['location_id']].drop_duplicates())
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
        print(f'Standardizing to population of {standardize_location_id}')
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
        loc_df_list = [df.loc[df['location_id'] == l] for l in df['location_id'].unique()]
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
        #print('Fix backcasting if we change nursing home observations (drop by name).')
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
            bc_df['Days'] = -(bc_df.index) - (start_rep - bc_rates[-1]) / bc_step
            # bc_df['Date'] = [df['Date'].min() - timedelta(days=x+1) for x in range(len(bc_df))]

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
    def _moving_3day_avg(day, df):
        # determine difference
        days = np.array([day-1, day, day+1])
        days = days[days >= 0]
        days = days[days <= df['Days'].max()]
        avg = df.loc[df['Days'].isin(days), rate_var].mean()

        return avg

    # get diffs
    avgs = [_moving_3day_avg(i, df) for i in df['Days']]
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
    def __init__(self, df: pd.DataFrame, data_version: str = 'best'):
        # expect passed in file to be `full_data.csv`
        self.data_version = data_version
        self.df = self._clean_up_dataset(df)
        
    def _clean_up_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # load hosptalizations
        hosp_df = pd.read_csv(
            f'/ihme/covid-19/snapshot-data/{self.data_version}/covid_onedrive/'\
            'location time series/locs_with_deaths_hosp_cumulative.csv',
            encoding='latin1'
        )
        hosp_df['Date'] = pd.to_datetime(hosp_df['date'], format='%d.%m.%Y')
        hosp_df['location_id'] = hosp_df['location_id'].apply(lambda x: int(x.split('/t')[-1]) if isinstance(x, str) else x)
        hosp_df = hosp_df.rename(index=str, columns={'hospitalizations':'Hospitalizations'})
        hosp_df = hosp_df[['location_id', 'Date', 'Hospitalizations']]
        df = df.merge(hosp_df, how='left')
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
        loc_df = pd.concat([moving_average(loc_df, smooth_var, reset_days=reset_days) for loc_df in loc_dfs])
        
        return loc_df
    
    def _average_over_last_days(self, df: pd.DataFrame, avg_var: str, mean_window: int = 3) -> pd.DataFrame:
        df['latest date'] = df.groupby('location_id', as_index=False)['Date'].transform(max)
        df['last three days'] = df['latest date'].apply(lambda x: x - timedelta(days=mean_window-1))
        df = df.loc[df['Date'] >= df['last three days']]
        df = df.groupby('location_id', as_index=False)[avg_var].mean()
        
        return df
    
    def _get_death_to_prior_indicator(self) -> pd.DataFrame:
        # smooth cases, prepare to match with deaths 8 days later
        case_df = self._smooth_data(self.df, 'ln(confirmed case rate)')
        case_df['Confirmed case rate'] = np.exp(case_df['ln(confirmed case rate)'])
        case_df['Date'] = case_df['Date'].apply(lambda x: x + timedelta(days=8))
        full_case_df = case_df[['location_id', 'Date', 'Confirmed case rate']].copy()
        case_df = case_df.loc[case_df['Confirmed'] > 0].reset_index(drop=True)

        # do the same thing with hospitalizations
        hosp_df = self._smooth_data(self.df.loc[~self.df['Hospitalizations'].isnull()], 
                                    'ln(hospitalization rate)',
                                    reset_days=True)
        hosp_df['Hospitalization rate'] = np.exp(hosp_df['ln(hospitalization rate)'])
        hosp_df['Date'] = hosp_df['Date'].apply(lambda x: x + timedelta(days=8))
        full_hosp_df = hosp_df[['location_id', 'Date', 'Hospitalization rate']].copy()
        hosp_df = hosp_df.loc[hosp_df['Hospitalizations'] > 0].reset_index(drop=True)

        # smooth deaths
        death_df = self._smooth_data(self.df, 'ln(death rate)')
        death_df['Death rate'] = np.exp(death_df['ln(death rate)'])
        full_death_df = death_df[['location_id', 'Date', 'Deaths', 'Death rate']].copy()
        death_df = death_df.loc[death_df['Deaths'] > 0].reset_index(drop=True)
        
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
        dhr_df = self._average_over_last_days(dhr_df, 'dhr lag8', 3)
        
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
        
        
        # call it ln(asdr) since we are only adding to locations in their own models
        df['ln(age-standardized death rate)'] = np.log(df['Death rate'])
        
        return df
    
    def produce_deaths(self) -> pd.DataFrame:
        # get ratio
        case_df, hosp_df, death_df, dcr_df, dhr_df = self._get_death_to_prior_indicator()
        
        # set limits on ratio based on number of deaths...
        #   - if >= 10, use 2.5th/97.5th of ratios in places with more than
        #     30 deaths (0.02, 0.2)
        #   - if < 10, use 10th/90th of that group (0.03, 0.15)
        death_df['last date'] = death_df.groupby('location_id', as_index=False)['Date'].transform(max)
        death_df = death_df.loc[death_df['Date'] == death_df['last date']]
        death_df = death_df[['location_id', 'last date', 'Deaths', 'Death rate']]
#         ratio_df = ratio_df.merge(death_df[['location_id', 'Deaths']])
#         ratio_df.loc[(ratio_df['Deaths'] >= 10) & (ratio_df['dcr lag8'] < 0.02), 'dcr lag8'] = 0.02
#         ratio_df.loc[(ratio_df['Deaths'] >= 10) & (ratio_df['dcr lag8'] > 0.2), 'dcr lag8'] = 0.2
#         ratio_df.loc[(ratio_df['Deaths'] < 10) & (ratio_df['dcr lag8'] < 0.03), 'dcr lag8'] = 0.03
#         ratio_df.loc[(ratio_df['Deaths'] < 10) & (ratio_df['dcr lag8'] > 0.15), 'dcr lag8'] = 0.15
#         del ratio_df['Deaths']
#         ratio_df['location_id'] = ratio_df['location_id'].astype(int)
        
        # apply ratio to get deaths (use <10 deaths floor as ratio for places without deaths thus far)
        dc_df = case_df.merge(dcr_df[['location_id', 'dcr lag8']], how='left')
        dh_df = hosp_df.merge(dhr_df[['location_id', 'dhr lag8']], how='left')
        dc_df['location_id'] = dc_df['location_id'].astype(int)
        dh_df['location_id'] = dh_df['location_id'].astype(int)
#         df['dcr lag8'] = df['dcr lag8'].fillna(0.03)
        dc_df['Death rate'] = dc_df['Confirmed case rate'] * dc_df['dcr lag8']
        dh_df['Death rate'] = dh_df['Hospitalization rate'] * dh_df['dhr lag8']
        
        # start daily deaths from last data point
        death_df['location_id'] = death_df['location_id'].astype(int)
        dc_df = self._stream_out_deaths(dc_df, death_df)
        dc_df = dc_df.rename(index=str, columns={'ln(age-standardized death rate)': 'from_cases'})
        dh_df = self._stream_out_deaths(dh_df, death_df)
        dh_df = dh_df.rename(index=str, columns={'ln(age-standardized death rate)': 'from_hospital'})
        df = dc_df[['location_id', 'Date', 'from_cases']].merge(
            dh_df[['location_id', 'Date', 'from_hospital']],
            how='outer'
        )
        
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        # TODO:structure is for diagnostic purposes, update
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        
        return dcr_df, dhr_df, df  # [['location_id', 'Date', 'ln(age-standardized death rate)']]
        