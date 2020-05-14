import sys
from functools import reduce
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict

from matplotlib.backends.backend_pdf import PdfPages

from front_end_loader import load_locations, load_cases_deaths_pop, load_testing
from smoother import smoother
from cdr_model import cdr_model

import warnings
warnings.simplefilter('ignore')


def find_missing_locations(df: pd.DataFrame, measure: str, loc_df: pd.DataFrame) -> List[int]:
    missing_list = list(set(loc_df['location_id']) - set(df['location_id'].unique()))
    if len(missing_list) > 0:
        print(f"Missing {measure} for "
              f"{';'.join(loc_df.loc[loc_df['location_id'].isin(missing_list), 'location_name'].to_list())}")
    
    return missing_list
    

def main(location_set_version_id: int, inputs_version: str, testing_version: str):
    # load all data we have
    loc_df = load_locations(location_set_version_id)
    case_df, death_df, pop_df = load_cases_deaths_pop(inputs_version)
    test_df = load_testing(testing_version)
    
    # identify locations for which we do not have all data
    missing_cases = find_missing_locations(case_df, 'cases', loc_df)
    missing_deaths = find_missing_locations(death_df, 'deaths', loc_df)
    missing_testing = find_missing_locations(test_df, 'testing', loc_df)
    missing_locations = list(set(missing_cases + missing_deaths + missing_testing))
    
    # combine data
    df = reduce(lambda x, y: pd.merge(x, y, how='outer'),
                [case_df[['location_id', 'Date', 'Confirmed case rate']],
                 test_df[['location_id', 'Date', 'Testing rate']],
                 death_df[['location_id', 'Date', 'Death rate']],
                 pop_df[['location_id', 'population']]])
    df = loc_df[['location_id', 'location_name']].merge(df)
    not_missing = ~df['location_id'].isin(missing_locations)
    df = df.loc[not_missing]    
    
    # smooth deaths and cases
    cumul_df = (df.groupby('location_id', as_index=False)
                .apply(lambda x: smoother(x, ['Confirmed case rate', 'Testing rate', 'Death rate'], 
                                          daily=False, log=False))
                .reset_index(drop=True))
    ln_cumul_df = (df.groupby('location_id', as_index=False)
                   .apply(lambda x: smoother(x, ['Confirmed case rate', 'Testing rate', 'Death rate'], 
                                             daily=False, log=True))
                   .reset_index(drop=True))
    daily_df = (df.groupby('location_id', as_index=False)
                .apply(lambda x: smoother(x, ['Confirmed case rate', 'Testing rate', 'Death rate'], 
                                          daily=True, log=False))
                .reset_index(drop=True))
    ln_daily_df = (df.groupby('location_id', as_index=False)
                   .apply(lambda x: smoother(x, ['Confirmed case rate', 'Testing rate', 'Death rate'], 
                                             daily=True, log=True))
                   .reset_index(drop=True))

    # run models
    with PdfPages('/ihme/homes/rmbarber/covid-19/alt_deaths/model_cumul.pdf') as pdf:
        cumul_df = (cumul_df.groupby('location_id', as_index=False)
                    .apply(lambda x: cdr_model(x, -np.inf, 
                                               daily=False, log=False,
                                               dep_var='Smoothed death rate', 
                                               indep_vars=['Smoothed confirmed case rate', 'Smoothed testing rate'],
                                               pdf=pdf))
                    .reset_index(drop=True))
    with PdfPages('/ihme/homes/rmbarber/covid-19/alt_deaths/model_ln_cumul.pdf') as pdf:
        ln_cumul_df = (ln_cumul_df.groupby('location_id', as_index=False)
                       .apply(lambda x: cdr_model(x, -np.inf, 
                                                  daily=False, log=True,
                                                  dep_var='Smoothed death rate', 
                                                  indep_vars=['Smoothed confirmed case rate', 'Smoothed testing rate'],
                                                  pdf=pdf))
                       .reset_index(drop=True))
    with PdfPages('/ihme/homes/rmbarber/covid-19/alt_deaths/model_daily.pdf') as pdf:
        daily_df = (daily_df.groupby('location_id', as_index=False)
                    .apply(lambda x: cdr_model(x, -np.inf, 
                                               daily=True, log=False,
                                               dep_var='Smoothed death rate', 
                                               indep_vars=['Smoothed confirmed case rate', 'Smoothed testing rate'],
                                               pdf=pdf))
                    .reset_index(drop=True))
    with PdfPages('/ihme/homes/rmbarber/covid-19/alt_deaths/model_ln_daily.pdf') as pdf:
        ln_daily_df = (ln_daily_df.groupby('location_id', as_index=False)
                       .apply(lambda x: cdr_model(x, -np.inf, 
                                                  daily=True, log=True,
                                                  dep_var='Smoothed death rate', 
                                                  indep_vars=['Smoothed confirmed case rate', 'Smoothed testing rate'],
                                                  pdf=pdf))
                       .reset_index(drop=True))
    #df.to_csv('/ihme/homes/rmbarber/covid-19/alt_deaths/model_ln_cumul.csv', index=False)
    
    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
