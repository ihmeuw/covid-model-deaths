import pandas as pd
from typing import Tuple
from db_queries import get_location_metadata


def load_locations(location_set_version_id: int) -> pd.DataFrame:
    df = get_location_metadata(location_set_id=111, location_set_version_id=location_set_version_id)
    most_detailed = df['most_detailed'] == 1
    keep_columns = ['location_id', 'location_name', 'path_to_top_parent']
    df = df.loc[most_detailed, keep_columns]
    
    return df


def fill_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('Date').set_index('Date')
    df = df.asfreq('D', method='pad').reset_index()
    
    return df

def load_cases_deaths_pop(inputs_version: str = 'best') -> Tuple[pd.DataFrame]:
    # read in dataframe
    full_df = pd.read_csv(f'/ihme/covid-19/model-inputs/{inputs_version}/full_data.csv')
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    full_df['location_id'] = full_df['location_id'].astype(int)
    
    # make sure we have case rates
    full_df['Confirmed case rate'] = full_df['Confirmed'] / full_df['population']
    
    # only keep where we have deaths and cases
    full_df = full_df[['location_id', 'Date', 'Confirmed case rate', 'Death rate', 'population']].reset_index(drop=True)
    full_df = full_df.sort_values(['location_id', 'Date']).reset_index(drop=True)
    
    # case data shifted 8 days in future
    case_df = full_df[['location_id', 'Date', 'Confirmed case rate']].copy()
    case_df['True date'] = case_df['Date']
    case_df['Date'] = case_df['Date'].apply(lambda x: x + pd.Timedelta(days=8))
    non_na = ~case_df['Confirmed case rate'].isnull()
    has_cases = case_df.groupby('location_id')['Confirmed case rate'].transform(max).astype(bool)
    case_df = case_df.loc[non_na & has_cases]
    case_df = case_df[['location_id', 'True date', 'Date', 'Confirmed case rate']].reset_index(drop=True)
    case_df = (case_df.groupby('location_id', as_index=False)
               .apply(lambda x: fill_dates(x))
               .reset_index(drop=True))
    
    # death data
    death_df = full_df[['location_id', 'Date', 'Death rate']].copy()
    non_na = ~death_df['Death rate'].isnull()
    has_deaths = death_df.groupby('location_id')['Death rate'].transform(max).astype(bool)
    death_df = death_df.loc[non_na & has_deaths]
    death_df = death_df.reset_index(drop=True)
    death_df = (death_df.groupby('location_id', as_index=False)
                .apply(lambda x: fill_dates(x))
                .reset_index(drop=True))
    
    # population data
    pop_df = full_df[['location_id', 'population']].drop_duplicates()
    pop_df = pop_df.reset_index(drop=True)
    
    return case_df, death_df, pop_df


def load_testing(testing_version: str = 'best') -> pd.DataFrame:
    # read in data
    df = pd.read_csv(f'/ihme/covid-19/testing-outputs/{testing_version}/forecast_raked_test_pc_simple.csv')
    df['Date'] = pd.to_datetime(df['date'])
    df['True date'] = df['Date']
    df['Date'] = df['Date'].apply(lambda x: x + pd.Timedelta(days=8))
    
    # keep real data
    observed = df['observed']
    non_na = ~df['test_pc'].isnull()
    has_tests = df.groupby('location_id')['test_pc'].transform(max).astype(bool)
    df = df.loc[observed & non_na & has_tests]
    
    # transform from daily to cumulative
    df = df.sort_values(['location_id', 'Date'])
    df['Testing rate'] = df.groupby('location_id')['test_pc'].cumsum()
    
    # keep columns we need
    df = df[['location_id', 'True date', 'Date', 'Testing rate']].reset_index(drop=True)
    df = (df.groupby('location_id', as_index=False)
          .apply(lambda x: fill_dates(x))
          .reset_index(drop=True))
    
    return df
