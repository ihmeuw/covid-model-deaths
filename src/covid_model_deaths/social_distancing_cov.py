from datetime import datetime, timedelta
from typing import List, Union

import numpy as np
import pandas as pd

# FIXME: Also defined in covid-inputs-etl.
RENAME = {
    'Mecklenburg-Western Pomerania': 'Mecklenburg-Vorpommern',
    'Principado de Asturias': 'Asturias',
    'Islas Baleares': 'Balearic Islands',
    'Islas Canarias': 'Canary Islands',
    'Castilla y Leon': 'Castile and Leon',
    'Cataluna': 'Catalonia',
    'Comunidad Valenciana': 'Valencian Community',
    'Comunidad de Madrid': 'Community of Madrid',
    'Region de Murcia': 'Murcia',
    'Comunidad Foral de Navarra': 'Navarre',
    'Pais Vasco': 'Basque Country'
}

# TODO: get in snapshot/model-inputs
EFFECT_FILE = '/ihme/covid-19/deaths/mobility_inputs/2020_04_14/effs_on_DL_GLavg_SG.csv'


class SocialDistCov:
    closure_cols = ['People instructed to stay at home',
                    'Educational facilities closed',
                    'Non-essential services closed (i.e., bars/restaurants)',
                    'Rationing of supplies and requsitioning of facilities',
                    'Travel severely limited',
                    'Major reprioritisation of healthcare services',
                    'Any Gathering Restrictions',
                    'Any Business Closures']
    closure_level_idx = [0, 1, 2, 4]

    def __init__(self, death_df: pd.DataFrame, date_df: pd.DataFrame = None, data_version: str = 'best'):
        # read in and format closure data
        # TODO: Move to etl
        self.closure_sheet = f'/ihme/covid-19/model-inputs/{data_version}/closure_criteria_sheet.xlsx'
        self.closure_df = self._process_closure_dataset()

        # use current date
        self.current_date = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d')

        # load threshold death rate
        self.thresh_df = self._get_threshold_date(death_df, date_df)

    # TODO: move to etl
    def _process_closure_dataset(self) -> pd.DataFrame:
        # load data, keep relevant rows/columns
        df = pd.read_excel(self.closure_sheet)

        # FIXME: there have been issues in the past with merge_name and
        #  country not being the same, such that the threshold merge fails.
        #  recognize this function's reliance on merge_name if debugging.
        # fix names
        df = df.rename(index=str, columns={'merge_name': 'Location', 'country': 'Country/Region'})
        df = df.loc[(df['notes and additional information'] != 'NOT') &
                    (~df['Location'].isnull()) & (~df['Country/Region'].isnull())]
        df.loc[df['Location'].isin(list(RENAME.keys())), 'Location'] = (
            df.loc[df['Location'].isin(list(RENAME.keys())), 'Location'].map(RENAME)
        )

        # rest of data
        df.loc[df['Country/Region'].isnull(), 'Country/Region'] = df['Location']
        df.loc[df['Country/Region'] == 'USA', 'Country/Region'] = 'United States of America'
        df = df.loc[~(df['location_id'].isnull()) & 
                    ~(df['Location'].isnull()) & 
                    ~(df['Country/Region'].isnull())]
        #df = df[['Location', 'Country/Region'] + self.closure_cols]
        
        # replace Wuhan location_id until we it is updated in the ETL
        df.loc[df['Location'] == 'Wuhan City, Hubei', 'location_id'] = -503002
        
        # just keep location_id as identifier
        df = df[['location_id'] + self.closure_cols]
        df['location_id'] = df['location_id'].astype(int)
        # convert datetime column
        for date_col in self.closure_cols:
            df[date_col] = df[date_col].apply(
                lambda x: datetime.strptime(x, '%d.%m.%Y') if isinstance(x, str) and x[0].isdigit() else np.nan
            )

        return df.reset_index(drop=True)

    @staticmethod
    def _get_threshold_date(df: pd.DataFrame, date_df: pd.DataFrame) -> pd.DataFrame:
        # walk back from the first real date to day 0 in the model
        df = df.loc[~df['Date'].isnull()].copy()
        df['first_date'] = (df
                            .groupby(['location_id', 'Location', 'Country/Region'], as_index=False)['Date']
                            .transform('min'))
        df = df.loc[df['Date'] == df['first_date']]
        df['threshold_date'] = df.apply(lambda x: x['Date'] - timedelta(days=np.round(x['Days'])), axis=1)

        # tack on mean date data
        df = df[['location_id', 'Location', 'Country/Region', 'threshold_date']].reset_index(drop=True)
        if date_df is not None:
            df = df.append(
                date_df.loc[~date_df['Location'].isin(df['Location'].unique().tolist()),
                            ['location_id', 'Location', 'Country/Region', 'threshold_date']]
            ).reset_index(drop=True)

        return df

    def _calc_peak_date(self, prob: str, R0_file: str):
        # get R0 == 1 file
        r0_df = pd.read_csv(R0_file)
        if prob == 'R0_35':
            r0_df['R0 date'] = pd.to_datetime(r0_df['p35_date'])
        elif prob == 'R0_50':
            r0_df['R0 date'] = pd.to_datetime(r0_df['p50_date'])
        elif prob == 'R0_65':
            r0_df['R0 date'] = pd.to_datetime(r0_df['p65_date'])

        # get days from threshold
        df = self.thresh_df.merge(r0_df[['location_id', 'R0 date']])

        # get predicted days from threshold to peak (will act directly on beta;
        # not actually composite like other covariates, should improve this)
        df['cov_1w'] = df.apply(lambda x: (x['R0 date'] - x['threshold_date']).days + 19, axis=1)
        df['cov_2w'] = np.nan
        df['cov_3w'] = np.nan
        df = df.loc[df['cov_1w'] > 0]

        return df[['location_id', 'Location', 'Country/Region', 'threshold_date', 'R0 date', 
                   'cov_1w', 'cov_2w', 'cov_3w']]

    def _calc_composite_empirical_weights(self, empirical_weight_source: str):
        # map of closure codes to names
        code_map = {'ci_sd1':'People instructed to stay at home',
                    'ci_sd2':'Educational facilities closed',
                    'ci_sd3':'Non-essential services closed (i.e., bars/restaurants)',
                    'ci_psd1':'Any Gathering Restrictions',
                    'ci_psd3':'Any Business Closures'}

        # load data, just keep average
        weight_df = pd.read_csv(EFFECT_FILE)
        weight_df = weight_df.loc[weight_df['statistic'] == 'mean']
        if empirical_weight_source == 'google':
            weight_df = weight_df.loc[weight_df['metric'] == 'Google_avg_of_retail_transit_workplace']
        elif empirical_weight_source == 'descartes':
            weight_df = weight_df.loc[weight_df['metric'] == 'Descartes_absolute_travel_distance']
        elif empirical_weight_source == 'safegraph':
            weight_df = weight_df.loc[weight_df['metric'] == 'Safegraph_time_outside_home']
        else:
            raise ValueError('Invalid `empirical_weight_source` provided.')

        # set to proportional reduction (i.e., positive, out of 1)
        weight_df[list(code_map.keys())] = weight_df[list(code_map.keys())].values

        # remove partial effect from full (will use these as compounding in weighting)
        weight_df['ci_sd1'] = weight_df['ci_sd1'] - weight_df['ci_psd1']
        weight_df['ci_sd3'] = weight_df['ci_sd3'] - weight_df['ci_psd3']
        weight_df = pd.melt(weight_df,
                            id_vars=['metric'],
                            value_vars=list(code_map.keys()),
                            var_name='closure_code',
                            value_name='effect')
        weight_df['closure_name'] = weight_df['closure_code'].map(code_map)
        weight_denom = weight_df['effect'].sum()
        weight_df['weight'] = weight_df['effect'] / weight_denom
        weight_dict = dict(zip(weight_df['closure_code'], weight_df['weight']))

        # get days from threshold
        df = self.thresh_df.merge(self.closure_df)
        df = df.loc[~df['threshold_date'].isnull()]
        for closure_code, closure_name in code_map.items():
            df[closure_code] = df.apply(lambda x: (x[closure_name] - x['threshold_date']).days, axis=1)

        # fill parial with full if it is null (i.e., since we are using them as compounding effects, full incorporates partial)
        # if both are null, obviously does nothing
        df.loc[df['ci_psd1'].isnull(), 'ci_psd1'] = df['ci_sd1']
        df.loc[df['ci_psd3'].isnull(), 'ci_psd3'] = df['ci_sd3']

        # fill nulls with 3 weeks
        for closure_code in code_map.keys():
            df.loc[df[closure_code].isnull(), closure_code] =  df.loc[df[closure_code].isnull()].apply(
                lambda x: (self.current_date - x['threshold_date']).days + 21, axis=1
            )

        # combine w/ weights
        df['composite_1w'] = np.nan
        df['composite_2w'] = np.nan
        df['composite_3w'] = (df[list(code_map.keys())] * np.array(list(weight_dict.values()))).sum(axis=1)

        return df[['location_id', 'Location', 'Country/Region', 'threshold_date']
                   + list(code_map.keys())
                   + ['composite_1w', 'composite_2w', 'composite_3w']]
    
    def _calc_composite_explicit_weights(self, weights: Union[List[int], List[float], np.ndarray]) -> pd.DataFrame:
        # scale weights
        if isinstance(weights, list):
            weights = np.array(weights)
        weights = weights / weights.sum()

        # how many levels are we interested in
        n_levels = weights.size
        assert n_levels == 3, 'Go back and check if this is != 3.'

        # get days from threshold
        df = self.thresh_df.merge(self.closure_df)
        for i in self.closure_level_idx:
            df[f'closure_id_{i}'] = df.apply(lambda x: (x[self.closure_cols[i]] - x['threshold_date']).days, axis=1)

        # get smallest [n_levels] day counts out of our list of closure dates
        for i in range(n_levels):
            df[f'closure_{i+1}'] = df.apply(
                lambda x: np.sort(
                    np.array(x[[f'closure_id_{j}' for j in self.closure_level_idx]].to_list())
                )[i],
                axis=1
            )
        closure_vars = [f'closure_{i+1}' for i in range(n_levels)]

        # 1 week
        # get composite
        df['composite_1w'] = (df[closure_vars] * weights).sum(axis=1)

        # adjust if missing
        for n_not_met in range(1, n_levels+1):
            df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met,
                   'composite_1w'] = df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met].apply(
                lambda x: x['composite_1w'] + (self.current_date + timedelta(days=7) - x['threshold_date']).days * np.flip(weights)[:n_not_met].sum(),
                axis=1
            )

        # 2 week
        # get composite
        df['composite_2w'] = (df[closure_vars] * weights).sum(axis=1)

        # adjust if missing
        for n_not_met in range(1, n_levels+1):
            df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met,
                   'composite_2w'] = df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met].apply(
                lambda x: x['composite_2w'] + (self.current_date + timedelta(days=14) - x['threshold_date']).days * np.flip(weights)[:n_not_met].sum(),
                axis=1
            )

        # 3 weeks
        # get composite
        df['composite_3w'] = (df[closure_vars] * weights).sum(axis=1)

        # adjust if missing
        for n_not_met in reversed(range(1, n_levels+1)):
            df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met,
                   'composite_3w'] = df.loc[df[closure_vars].isnull().sum(axis=1) == n_not_met].apply(
                lambda x: x['composite_3w'] + (self.current_date + timedelta(days=21) - x['threshold_date']).days * np.flip(weights)[:n_not_met].sum(),
                axis=1
            )

        return df[['Location', 'Country/Region', 'threshold_date']
                  + closure_vars
                  + ['composite_1w', 'composite_2w', 'composite_3w']]

    # FIXME: mutable default
    def get_cov_df(self, weights: Union[List[int], List[float], np.ndarray] = [1, 1, 1], k: int = 20,
                   empirical_weight_source: str = None, R0_file: str = None):
        # get composites
        if empirical_weight_source in ['google', 'descartes', 'safegraph']:
            df = self._calc_composite_empirical_weights(empirical_weight_source)
        elif empirical_weight_source in ['R0_35', 'R0_50', 'R0_65']:
            df = self._calc_peak_date(empirical_weight_source, R0_file)
        else:
            df = self._calc_composite_explicit_weights(weights)

        # scale to Wuhan
        if 'cov_1w' not in df.columns:
            wuhan_score_1w = df.loc[df['Location'] == 'Wuhan City, Hubei', 'composite_1w'].item()
            wuhan_score_2w = df.loc[df['Location'] == 'Wuhan City, Hubei', 'composite_2w'].item()
            wuhan_score_3w = df.loc[df['Location'] == 'Wuhan City, Hubei', 'composite_3w'].item()
            df['cov_1w'] = (df['composite_1w'] + k) / (wuhan_score_1w + k)
            df['cov_2w'] = (df['composite_2w'] + k) / (wuhan_score_2w + k)
            df['cov_3w'] = (df['composite_3w'] + k) / (wuhan_score_3w + k)

        return df.loc[~df['cov_3w'].isnull()].reset_index(drop=True)
