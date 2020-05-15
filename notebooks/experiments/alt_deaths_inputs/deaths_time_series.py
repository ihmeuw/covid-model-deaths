import sys
from functools import reduce
import pandas as pd
import numpy as np
import dill as pickle
from typing import List, Dict

from matplotlib.backends.backend_pdf import PdfPages

from front_end_loader import load_locations, load_cases_deaths_pop, load_testing
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
    
    # fit model
    with PdfPages('/ihme/homes/rmbarber/covid-19/alt_deaths/model_results.pdf') as pdf:
        df = (df.groupby('location_id', as_index=False)
              .apply(lambda x: cdr_model(x, -15, 
                                         daily=False, log=True, smooth_results=True, 
                                         death_var='Death rate',
                                         case_var='Confirmed case rate',
                                         test_var='Testing rate',
                                         pdf=pdf))
              .reset_index(drop=True))
    
    # save output
    df.to_csv('/ihme/homes/rmbarber/covid-19/alt_deaths/model_results.csv', index=False)
    
    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
