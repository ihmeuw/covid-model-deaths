import functools
import multiprocessing

import numpy as np
import pandas as pd
import tqdm


def clean_data(df):
    
    df['case_count'] = df['Confirmed']
    assert not np.any(df.case_count < 0)
    df['death_count'] = df['Deaths']
    assert not np.any(df.death_count < 0)
    df['case_rate'] = df['Confirmed case rate']
    df['death_rate'] = df['Death rate']
    df['date'] = df['Date'].map(pd.Timestamp)
    assert df['date'].min() == pd.Timestamp('2019-12-31')
    
    return df


def days_from_X_cases_to_Y_deaths(df,
                                  case_count_threshold=None, ln_case_rate_threshold=None,
                                  death_count_threshold=None, ln_death_rate_threshold=None):
    """find distribution of number of days from X cases to Y deaths
    
    Parameters
    ----------
    df : pd.DataFrame of outbreak data, including columns case_count, case_rate,
         death_count, death_rate, location, and date
    case_count_threshold : int, optional
    ln_case_rate_threshold : float, optional
    death_count_threshold : int, optional
    ln_death_rate_threshold : float, optional
    
    Results
    -------
    Returns pd.Series indexed by locations, containing the number of days after reaching 
    the case threshold that this location reached the death threshold
    
    Notes
    -----
    Exactly one of case_count_threshold and ln_case_rate_threshold must be specified; same
    with death_count_threshold and ln_death_rate_threshold
    """
    
    if case_count_threshold != None:
        assert ln_case_rate_threshold == None, 'Only one case threshold should be specified'
        day_X_cases = df[df['case_count'] >= case_count_threshold].groupby('location_id').date.min()
    else:
        case_rate_threshold = np.exp(ln_case_rate_threshold)
        day_X_cases = df[df['case_rate'] >= case_rate_threshold].groupby('location_id').date.min()
        
    if death_count_threshold != None:
        assert ln_death_rate_threshold == None, 'Only one death threshold should be specified'
        day_Y_deaths = df[df['death_count'] >= death_count_threshold].groupby('location_id').date.min()
    else:
        death_rate_threshold = np.exp(ln_death_rate_threshold)
        day_Y_deaths = df[df['death_rate'] >= death_rate_threshold].groupby('location_id').date.min()

    wait_times = (day_Y_deaths - day_X_cases) / pd.Timedelta(days=1)
    return wait_times.dropna().sort_values()


def random_delta_days(waits):
    """Get a random time delta"""
    mu = waits.mean()
    std = waits.std()
    random_wait = np.random.normal(mu, std)
    if random_wait < 1:
        random_wait = 1-random_wait
    return pd.Timedelta(days=np.round(random_wait))


def location_specific_death_threshold_date(df, location_id, ln_death_rate_threshold):
    results = pd.Series({'location_id':location_id})

    # find most recent case date for this location_id
    df_loc = df[df.location_id == location_id].sort_values('date', ascending=False)
    results['case_date'] = df_loc.iloc[0]['date']
    results['case_count'] = df_loc.iloc[0]['case_count']
    results['population'] = df_loc.iloc[0]['population']

    # now find the date when this location will/did cross the death rate threshold
    results['ln_death_rate_threshold'] = ln_death_rate_threshold
    death_rate_threshold = np.exp(ln_death_rate_threshold)
    
    # if this location has already crossed death_rate_threshold, add date on which it did
    if df_loc['death_rate'].max() >= death_rate_threshold:
        results['threshold_reached'] = True
        rows = df_loc['death_rate'] >= death_rate_threshold
        threshold_death_date = df_loc[rows]['date'].min()
        for i in range(1000):
            results[f'death_date_draw_{i:03d}'] = threshold_death_date
    else: # otherwise, sample from distribution
        results['threshold_reached'] = False
        observed_waits = days_from_X_cases_to_Y_deaths(df, case_count_threshold=results['case_count'],
                                                       ln_death_rate_threshold=ln_death_rate_threshold)
        # retain only positive waits, since we believe deaths are being reported accurately
        retained_waits = observed_waits[observed_waits > 0]
        for i in range(1000):
            results[f'death_date_draw_{i:03d}'] = (results['case_date']
                                                    + random_delta_days(retained_waits))
    return results

def try_location_specific_death_threshold_rate(location_id, df, ln_death_rate_threshold):
    try:
        return location_specific_death_threshold_date(df, location_id, ln_death_rate_threshold)
    except Exception as e:
        print(e)
        print(location_id, " failed")
        raise e


def impute_death_threshold(df,
                           location_list,
                           ln_death_rate_threshold=-15):
    """
    Run whole function on df for locations specified in location_list
    Data date added as a column in results df
    Return draws of date that death threshold will be reached by country

    """
    # step 1 - load in data
    df = clean_data(df)

    # step 2 - make sure location_list is ok
    assert set(location_list).issubset(set(df['location_id']))

    # step 3 - run functions on df
    np.random.seed(12345)
    _combiner = functools.partial(try_location_specific_death_threshold_rate,
                                 df=df, 
                                 ln_death_rate_threshold=ln_death_rate_threshold)
    with multiprocessing.Pool(20) as p:
        results = list(tqdm.tqdm(p.imap(_combiner, location_list), total=len(location_list)))
    
    results = pd.DataFrame(results)

    return results
