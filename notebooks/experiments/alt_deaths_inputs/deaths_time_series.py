import os
import sys
from functools import reduce
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict

from matplotlib.backends.backend_pdf import PdfPages

from front_end_loader import load_locations, load_cases_deaths_pop, load_testing
from cfr_model import cfr_model, synthesize_time_series

import warnings
warnings.simplefilter('ignore')


def holdout_days(df: pd.DataFrame, n_holdout_days: int) -> pd.DataFrame:
    df = df.copy()
    df['last_date'] = df.groupby('location_id')['Date'].transform(max)
    keep_idx = df.apply(lambda x: x['Date'] <= x['last_date'] - pd.Timedelta(days=n_holdout_days), axis=1)
    df = df.loc[keep_idx].reset_index(drop=True)
    del df['last_date']
    
    return df


def find_missing_locations(df: pd.DataFrame, measure: str, loc_df: pd.DataFrame) -> List[int]:
    missing_list = list(set(loc_df['location_id']) - set(df['location_id'].unique()))
    if len(missing_list) > 0:
        print(f"Missing {measure} for "
              f"{';'.join(loc_df.loc[loc_df['location_id'].isin(missing_list), 'location_name'].to_list())}")
    
    return missing_list
    

def main(location_set_version_id: int, inputs_version: str, testing_version: str,
         run_label: str, n_holdout_days: int = 0):
    # set up out dir
    out_dir = f'/ihme/covid-19/deaths/dev/{run_label}'
    if os.path.exists(out_dir):
        #raise ValueError('Directory already exists.')
        pass
    else:
        os.mkdir(out_dir)
    
    # load all data we have
    loc_df = load_locations(location_set_version_id)
    case_df, death_df, pop_df = load_cases_deaths_pop(inputs_version)
    test_df = load_testing(testing_version)
    
    # drop days of data as specified
    case_df = holdout_days(case_df, n_holdout_days)
    death_df = holdout_days(death_df, n_holdout_days)
    test_df = holdout_days(test_df, n_holdout_days)
    
    # identify locations for which we do not have all data
    missing_cases = find_missing_locations(case_df, 'cases', loc_df)
    missing_deaths = find_missing_locations(death_df, 'deaths', loc_df)
    missing_testing = find_missing_locations(test_df, 'testing', loc_df)
    missing_locations = list(set(missing_cases + missing_deaths + missing_testing))
    
    # add some poorly behaving locations to missing list
    # Assam (4843); 
    missing_locations += [4843]
    
    # combine data
    df = reduce(lambda x, y: pd.merge(x, y, how='outer'),
                [case_df[['location_id', 'Date', 'Confirmed case rate']],
                 test_df[['location_id', 'Date', 'Testing rate']],
                 death_df[['location_id', 'Date', 'Death rate']],
                 pop_df[['location_id', 'population']]])
    df = loc_df[['location_id', 'location_name']].merge(df)
    not_missing = ~df['location_id'].isin(missing_locations)
    df = df.loc[not_missing]  
    
    # fit model
    np.random.seed(15243)
    var_dict = {'death_var':'Death rate',
                'case_var':'Confirmed case rate',
                'test_var':'Testing rate'}
    df = (df.groupby('location_id', as_index=False)
         .apply(lambda x: cfr_model(x, 
                                    deaths_threshold=max(1,
                                                         int((x['Death rate']*x['population']).max()*0.01)), 
                                    daily=False, log=True, 
                                    **var_dict))
         .reset_index(drop=True))

    # fit spline to output
    with PdfPages(f'{out_dir}/model_results.pdf') as pdf:
        draw_df = (df.groupby('location_id', as_index=False)
                   .apply(lambda x: synthesize_time_series(
                       x, 
                       daily=True, log=True,
                       pdf=pdf, 
                       **var_dict
                   ))
                   .reset_index(drop=True))

    # save output
    df.to_csv(f'{out_dir}/model_data.csv', index=False)
    draw_df.to_csv(f'{out_dir}/model_results.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
