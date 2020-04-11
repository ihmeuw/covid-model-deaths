import os

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# from utilities import CompareModelDeaths
from .compare_moving_average import CompareAveragingModelDeaths
from seaborn import set_style

set_style("whitegrid")
COLUMNS = ['location_id', 'location', 'date', 'observed']
DRAWS = ['draw_{i}'.format(i=i) for i in np.arange(1000)]


def get_previous_date(n_diff: int):
    '''Get the date for previous `n_diff` days.'''
    previous_date = datetime.today() - timedelta(n_diff)
    date_str = previous_date.strftime("%Y_%m_%d").replace('-', '_')
    return date_str


def get_filepath(indir, date_dir, location):
    '''Return file path for death data'''
    if location == 'US':
        filepath = os.path.join(indir, date_dir, "state_data.csv")
    elif location == 'Europe':
        filepath = os.path.join(indir, date_dir, "euro_data.csv")
    else:
        raise (f"{location} does not exist!")
    return filepath


def get_previous_filepaths(indir: str, location: str, n_models=3):
    '''`location` should be `US` or `Europe` for now.'''
    if location not in ['US', 'Europe']:
        raise (f"{location} is not US or Europe")

    lst_paths = []
    for n_diff in range(0, 100):
        prev_date = get_previous_date(n_diff)
        prev_dir = f"{prev_date}_{location}"
        prev_filepath = get_filepath(indir, prev_dir, location)

        # If previous day's predictions don't exist,
        # then use the day before's predictions
        if os.path.exists(prev_filepath):
            lst_paths.append(prev_filepath)
        else:
            continue

        # Select up to `n_models` predictions; including today's run.
        if len(lst_paths) == n_models:
            break
        else:
            continue
    return lst_paths


def get_daily_deaths(df):
    '''Get daily deaths from cumulative deaths.'''
    df = df.sort_values(COLUMNS)
    tmp_cols = df[['date', 'observed']]
    df = (df.groupby(['location_id', 'location'])[DRAWS]
          .apply(lambda x: x.shift(0) - x.shift(1).fillna(0)))
    # Add back `date` and `observed` columns
    df = df.reset_index(['location_id', 'location']).join(tmp_cols)[COLUMNS + DRAWS]
    return df


def average_model_outputs(lst_paths, model_1, indir, location):
    '''Average over model outputs'''
    lst_df_prev = []
    for inpath in lst_paths:
        df_prev = pd.read_csv(inpath)
        # Some European runs don't have 1000 draws.
        if 'draw_999' not in df_prev.columns:
            df_prev['draw_999'] = df_prev['draw_998']
        # All observed daily values need to be averaged. Comment out the following line.
        # df_prev = df_prev.loc[df_prev.observed == False]
        df_prev = get_daily_deaths(df_prev)
        lst_df_prev.append(df_prev)
    mean_df = (pd.concat(lst_df_prev)
               .groupby(COLUMNS)
               .mean().reset_index())

    ## To refactor
    # mean_df['Date'] = (mean_df['date']
    #     .map(lambda x: datetime.strptime(x, '%Y-%m-%d')
    #      if isinstance(x, str) else np.nan))
    # # Drop rows where observed is false and where the date is before today
    # # because these rows were predicted in earlier models.
    # mean_df = mean_df.loc[~((mean_df.observed == False)
    #                       & (mean_df.Date <= datetime.today() - timedelta(1)))]
    # mean_df = mean_df.drop('Date', axis=1)

    # df = get_today_data(indir, location)
    df = pd.read_csv(model_1)
    mean_df = mean_df.merge(df[COLUMNS])
    tmp_cols = mean_df[COLUMNS]

    # Get the cumulative deaths from daily deaths.
    mean_df = (mean_df.groupby(['location_id', 'location'])[DRAWS]
        .cumsum().join(tmp_cols)[COLUMNS + DRAWS])
    return mean_df


def get_today_data(indir, location):
    today_date = datetime.today().strftime("%Y_%m_%d").replace('-', '_')
    today_dir = f"{today_date}_{location}"
    # Specify `today_dir` for test purpose.
    # today_dir = "2020_04_06_US"
    # today_dir = "2020_04_06_Europe"
    today_filepath = get_filepath(indir, today_dir, location)
    df = pd.read_csv(today_filepath)
    return df


def append_observed_outputs(model_1, mean_df, location):
    # today_date = datetime.today().strftime("%Y_%m_%d").replace('-', '_')
    # today_dir = f"{today_date}_{location}"
    # # Specify `today_dir` for test purpose.
    # today_dir = "2020_04_06_US"
    # # today_dir = "2020_04_06_Europe"
    # today_filepath = get_filepath(indir, today_dir, location)
    # df = pd.read_csv(today_filepath)

    # df = get_today_data(indir, location)
    df = pd.read_csv(model_1)
    orig_len = len(df)
    mean_df = mean_df.merge(df[COLUMNS])

    df = df.loc[df.observed == True].append(mean_df.loc[mean_df.observed == False])
    df = (df.sort_values(COLUMNS)
          .drop_duplicates(COLUMNS))
    if orig_len != len(df):
        raise ValueError('Original data and averaged data are not same length.')
    return df


def moving_average_predictions(location, indir="/ihme/covid-19/deaths/prod",
                               specified=False, model_1=None, model_2=None, model_3=None):
    '''Average the predictions for recent 3 runs.'''
    if not specified:
        lst_paths = get_previous_filepaths(indir, location)
    else:
        lst_paths = [model_1, model_2, model_3]
    print("Averaging over the following files: ", lst_paths)
    mean_df = average_model_outputs(lst_paths, model_1, indir, location)
    full_df = append_observed_outputs(model_1, mean_df, location)
    return full_df


def plot_moving_average(location, indir="/ihme/covid-19/deaths/prod"):
    '''Plot averaged predictions.'''
    if location == 'Europe':
        raw_draw_path = '/ihme/covid-19/deaths/prod/2020_04_06_Europe/euro_data.csv'
        average_draw_path = '/ihme/covid-19/deaths/prod/2020_04_06_Europe_test/past_avg_state_data.csv'
        yesterday_draw_path = '/ihme/covid-19/deaths/prod/2020_04_05_Europe/euro_data.csv'
        before_yesterday_draw_path = '/ihme/covid-19/deaths/prod/2020_04_04_Europe/euro_data.csv'

        plotter = CompareAveragingModelDeaths(
            raw_draw_path=raw_draw_path,
            average_draw_path=average_draw_path,
            yesterday_draw_path=yesterday_draw_path,
            before_yesterday_draw_path=before_yesterday_draw_path
        )
        plotter.make_some_pictures(f'/ihme/covid-19/deaths/prod/2020_04_06_Europe_test/moving_average_compare.pdf',
                                   'Europe')
    elif location == 'US':
        raw_draw_path = '/ihme/covid-19/deaths/prod/2020_04_06_US/state_data.csv'
        average_draw_path = '/ihme/covid-19/deaths/prod/2020_04_06_US_test/past_avg_state_data.csv'
        yesterday_draw_path = '/ihme/covid-19/deaths/prod/2020_04_05_US/state_data.csv'
        before_yesterday_draw_path = '/ihme/covid-19/deaths/prod/2020_04_04_US/state_data.csv'

        plotter = CompareAveragingModelDeaths(
            raw_draw_path=raw_draw_path,
            average_draw_path=average_draw_path,
            yesterday_draw_path=yesterday_draw_path,
            before_yesterday_draw_path=before_yesterday_draw_path
        )
        plotter.make_some_pictures(f'/ihme/covid-19/deaths/prod/2020_04_06_US/moving_average_compare.pdf',
                                   'United States of America')


if __name__ == '__main__':
    # indir = "/ihme/covid-19/deaths/prod"
    # df_europe = moving_average_predictions('Europe',
    #     model_1='/ihme/covid-19/deaths/prod/2020_04_06_Europe/euro_data.csv',
    #     model_2='/ihme/covid-19/deaths/prod/2020_04_05_Europe/euro_data.csv',
    #     model_3='/ihme/covid-19/deaths/prod/2020_04_04_Europe/euro_data.csv')
    # df_europe.to_csv('/ihme/covid-19/deaths/prod/2020_04_06_Europe_test/past_avg_state_data.csv', index=False)

    df_us = moving_average_predictions('US',
                                       model_1='/ihme/covid-19/deaths/prod/2020_04_06_US/state_data.csv',
                                       model_2='/ihme/covid-19/deaths/prod/2020_04_05_US/state_data.csv',
                                       model_3='/ihme/covid-19/deaths/prod/2020_04_04_US/state_data.csv')
    df_us.to_csv('/ihme/covid-19/deaths/prod/2020_04_06_US_test/past_avg_state_data.csv', index=False)

    plot_moving_average('US')
