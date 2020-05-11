import numpy as np
import pandas as pd

from covid_model_deaths.globals import COLUMNS
from covid_model_deaths.data import drop_lagged_reports_by_location, add_moving_average_rates


class LeadingIndicator:
    def __init__(self, full_df: pd.DataFrame, data_version: str):
        # expect passed in file to be `full_data.csv`
        self.data_version = data_version
        self.full_df = self._clean_up_dataset(full_df)
        
    def _to_daily(self, df: pd.DataFrame, cum_var: str, daily_var: str) -> pd.DataFrame:
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        first_day = df['Date'] == df.groupby('location_id').Date.transform(min)
        delta_values = df[cum_var].values[1:] - df[cum_var].values[:-1]
        delta_values = delta_values[~first_day.values[1:]]
        df.loc[first_day, daily_var] = df[cum_var]
        df.loc[~first_day, daily_var] = delta_values
        
        return df
    
    def _to_cumulative(self, df: pd.DataFrame, daily_var: str, cum_var: str) -> pd.DataFrame:
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        df[cum_var] = df.groupby('location_id', as_index=False)[daily_var].cumsum()
        
        return df

    def _tests_per_capita(self, pop_df: pd.DataFrame) -> pd.DataFrame:
        # load testing data from snapshot
        us_df = pd.read_csv(
            f'/ihme/covid-19/snapshot-data/{self.data_version}/covid_onedrive/Testing/us_states_tests.csv')
        us_df['Date'] = pd.to_datetime(us_df['date'], format='%d.%m.%Y')
        us_df = us_df.rename(index=str, columns={'totaltestresults': 'Tests'})
        g_df = pd.read_csv(
            f'/ihme/covid-19/snapshot-data/{self.data_version}/covid_onedrive/Testing/global_admin0_tests.csv')
        g_df['Date'] = pd.to_datetime(g_df['date'], format='%d.%m.%Y')
        g_df = g_df.rename(index=str, columns={'total_tests': 'Tests'})
        df = us_df[['location_id', 'Date', 'Tests']].append(g_df[['location_id', 'Date', 'Tests']])
        # df = pd.read_csv('/home/j/temp/kcausey/covid19/test_prop/data_smooth_4_27_global.csv')
        # df = df.rename(index=str, columns={'daily_total':'Tests'})
        # df['Date'] = pd.to_datetime(df['date'])

        # format and get testing rate
        df = df.loc[(~df['location_id'].isnull()) & (~df['Tests'].isnull())]
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        # df['Tests'] = df.groupby('location_id', as_index=False)['Tests'].cumsum()
        df = df.merge(pop_df)
        df['Testing rate'] = df['Tests'] / df['population']
        
        # smooth
        df['ln(testing rate)'] = np.log(df['Testing rate'])
        df.loc[df['Tests'] == 0, 'ln(testing rate)'] = np.log(0.1 / df['population'])
        df['day0'] = df.groupby('location_id', as_index=False)['Date'].transform(min)
        df['Days'] = df.apply(lambda x: (x['Date'] - x['day0']).days, axis=1)
        df['location_id'] = df['location_id'].astype(int)

        return df[['location_id', 'Date', 'Days', 'Tests', 'Testing rate', 'ln(testing rate)']]

    def _account_for_positivity(self, t1: float, t2: float,
                                c1: float, c2: float,
                                logit_pos_int: float = -8.05,
                                logit_pos_logit_test: float = -0.78) -> float:
        logit = lambda x: np.log(x / (1 - x))
        expit = lambda x: 1 / (1 + np.exp(-x))

        p1 = expit(logit_pos_int + logit_pos_logit_test * logit(t1))
        p2 = expit(logit_pos_int + logit_pos_logit_test * logit(t2))
        increase_from_testing = (t2 * p2 - t1 * p1) / (t1 * p1)
        increase_from_testing = max(0, increase_from_testing)

        excess_reporting = ((c2 - c1) / c1) - increase_from_testing

        cases = c1 * (1 + excess_reporting)

        if cases < 0:
            cases = 0

        return cases

    def _control_for_testing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df = df.sort_values('Date').reset_index(drop=True)
        df = self._to_daily(df, 'Testing rate', 'Daily testing rate')
        df = self._to_daily(df, 'Confirmed case rate', 'Daily case rate')
        
        # keep last 8 days, get avg daily cases and testing from 8-10, then just keep 8
        future_df = df.loc[df['Date'] >= df['Date'].max() - pd.Timedelta(days=8)]

        # use case data if it is less than 3 days behind
        if future_df['Testing rate'].isnull().sum() < 3:
            future_df = future_df.sort_values('Date').reset_index(drop=True)
            start_cases = future_df['Daily case rate'][0]
            start_tests = future_df['Daily testing rate'][0]
            future_df = future_df.loc[~future_df['Testing rate'].isnull()]

            future_df['Daily case rate'] = future_df.apply(
                lambda x: self._account_for_positivity(start_tests, x['Daily testing rate'],
                                                       start_cases, x['Daily case rate']),
                axis=1
            )
            future_df['Daily case rate'][0] = future_df['Confirmed case rate'][0]
            future_df = self._to_cumulative(future_df, 'Daily case rate', 'Confirmed case rate')
            df = pd.concat([df.loc[df['Date'] < df['Date'].max() - pd.Timedelta(days=8)],
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

    def _average_over_last_days(self, df: pd.DataFrame, avg_var: str, mean_window: int = 3) -> pd.DataFrame:
        df = df.copy()
        df['latest date'] = df.groupby('location_id', as_index=False)['Date'].transform(max)
        df['last days'] = df['latest date'].apply(lambda x: x - pd.Timedelta(days=mean_window - 1))
        df = df.loc[df['Date'] >= df['last days']]
        df = df.groupby('location_id', as_index=False)[avg_var].mean()

        return df

    def _get_death_to_prior_indicator(self):
        # smooth cases, prepare to match with deaths 8 days later
        case_df = drop_lagged_reports_by_location(self.full_df, 'Confirmed')
        case_df = (case_df
                   .loc[case_df.groupby('location_id', as_index=False)['Confirmed'].transform(max).values > 0]
                   .reset_index(drop=True))
        case_df = add_moving_average_rates(case_df, 'ln(confirmed case rate)', -np.inf)
        case_df['Confirmed case rate'] = np.exp(case_df['ln(confirmed case rate)'])
        case_df['Date'] = case_df['Date'].apply(lambda x: x + pd.Timedelta(days=8))
        
        # adjust last 8 days of cases based on changes in testing over that time
        test_df = self._tests_per_capita(case_df[['location_id', 'population']].drop_duplicates())
        test_df = drop_lagged_reports_by_location(test_df, 'Tests')
        test_df = (test_df
                   .loc[test_df.groupby('location_id', as_index=False)['Tests'].transform(max).values > 0]
                   .reset_index(drop=True))
        test_df = add_moving_average_rates(test_df, 'ln(testing rate)', -np.inf)
        test_df['Testing rate'] = np.exp(test_df['ln(testing rate)'])
        test_df['Date'] = test_df['Date'].apply(lambda x: x + pd.Timedelta(days=8))
        case_df = case_df.merge(test_df[['location_id', 'Date', 'Testing rate']], how='left')
        case_df = pd.concat(
            [self._control_for_testing(case_df.loc[case_df['location_id'] == l]) for l in case_df.location_id.unique()]
        )
        full_case_df = case_df[['location_id', 'Date', 'Confirmed case rate']].copy()
        
        hosp_df = self.full_df.loc[~self.full_df['Hospitalizations'].isnull()]
        # do the same thing with hospitalizations (not present for all locs, so subset)
        hosp_df = drop_lagged_reports_by_location(hosp_df, 'Hospitalizations')
        hosp_df = (hosp_df
                   .loc[hosp_df.groupby('location_id', as_index=False)['Hospitalizations'].transform(max).values > 0]
                   .reset_index(drop=True))
        hosp_df = add_moving_average_rates(hosp_df, 'ln(hospitalization rate)', -np.inf)
        hosp_df['Hospitalization rate'] = np.exp(hosp_df['ln(hospitalization rate)'])
        hosp_df['Date'] = hosp_df['Date'].apply(lambda x: x + pd.Timedelta(days=8))
        full_hosp_df = hosp_df[['location_id', 'Date', 'Hospitalization rate']].copy()
        # hosp_df = hosp_df.loc[hosp_df['Hospitalizations'] > 0].reset_index(drop=True)

        # smooth deaths
        death_df = drop_lagged_reports_by_location(self.full_df, 'Deaths')
        death_df = (death_df
                    .loc[death_df.groupby('location_id', as_index=False)['Deaths'].transform(max).values > 0]
                    .reset_index(drop=True))
        death_df = add_moving_average_rates(death_df, 'ln(death rate)', -np.inf)
        death_df['Death rate'] = np.exp(death_df['ln(death rate)'])
        full_death_df = death_df[['location_id', 'Date', 'Deaths', 'Death rate']].copy()
        # death_df = death_df.loc[death_df['Deaths'] > 0].reset_index(drop=True)

        # calc ratios by day
        death_df = self._to_daily(death_df, 'Death rate', 'Daily death rate')
        case_df = self._to_daily(case_df, 'Confirmed case rate', 'Daily case rate')
        hosp_df = self._to_daily(hosp_df, 'Hospitalization rate', 'Daily hospitalization rate')
        dcr_df = death_df[['location_id', 'Date', 'Daily death rate']].merge(
            case_df[['location_id', 'Date', 'Daily case rate']]
        )
        dhr_df = death_df[['location_id', 'Date', 'Daily death rate']].merge(
            hosp_df[['location_id', 'Date', 'Daily hospitalization rate']]
        )
        dcr_df['dcr lag8'] = dcr_df['Daily death rate'] / dcr_df['Daily case rate']
        dhr_df['dhr lag8'] = dhr_df['Daily death rate'] / dhr_df['Daily hospitalization rate']

        # average ratios of ~last 3 days~ -> just use last day, is smoother
        dcr_df = self._average_over_last_days(dcr_df, 'dcr lag8', 1)
        if not dhr_df.empty:
            dhr_df = self._average_over_last_days(dhr_df, 'dhr lag8', 1)
        else:
            dhr_df = dhr_df[['location_id', 'dhr lag8']]

        return full_case_df, full_hosp_df, full_death_df, dcr_df, dhr_df

    def _stream_out_deaths(self, df: pd.DataFrame, death_df: pd.DataFrame) -> pd.DataFrame:
        # fix to last day of observed deaths
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        df = df[['location_id', 'Date', 'Daily death rate']].merge(death_df[['location_id', 'last date', 'Death rate']])
        df = df.loc[
            df['Date'] > df['last date']]  # will exclude days where diff is one locations last day and anothers first
        df = df.sort_values(['location_id', 'Date']).reset_index(drop=True)
        df['Death rate'] = df.groupby('location_id', as_index=False)['Daily death rate'].cumsum().values + \
                           df[['Death rate']].values

        return df

    def _combine_data(self, df: pd.DataFrame, rate_var: str, ratio_var: str,
                      ratio_df: pd.DataFrame, na_fill: float,
                      combine_var: str) -> pd.DataFrame:
        df = df.merge(ratio_df, how='left')
        df['location_id'] = df['location_id'].astype(int)
        df[ratio_var] = df[ratio_var].fillna(na_fill)
        df[combine_var] = df[rate_var] * df[ratio_var]
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

    def produce_deaths(self):
        # get ratio
        case_df, hosp_df, death_df, dcr_df, dhr_df = self._get_death_to_prior_indicator()

        # get last day of observed deaths and smoothed death rate
        death_df['last date'] = death_df.groupby('location_id', as_index=False)['Date'].transform(max)
        death_df = death_df.loc[death_df['Date'] == death_df['last date']]
        death_df = death_df[['location_id', 'last date', 'Deaths', 'Death rate']]

        # apply ratio to get deaths
        case_df = self._to_daily(case_df, 'Confirmed case rate', 'Daily case rate')
        hosp_df = self._to_daily(hosp_df, 'Hospitalization rate', 'Daily hospitalization rate')
        dc_df = self._combine_data(case_df, 'Daily case rate', 'dcr lag8',
                                   dcr_df[['location_id', 'dcr lag8']], np.nan,  # 0.03,
                                   combine_var='Daily death rate')
        dh_df = self._combine_data(hosp_df, 'Daily hospitalization rate', 'dhr lag8',
                                   dhr_df[['location_id', 'dhr lag8']], np.nan,  # 0.15,
                                   combine_var='Daily death rate')

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
