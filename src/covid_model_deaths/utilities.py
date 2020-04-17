import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import time



# TODO: Document better.  These are about the mix of social distancing
#  covariates.
MOBILITY_SOURCES = ['google', 'descartes', 'safegraph']
# COV_SETTINGS = [('equal', [1, 1, 1]),
#                 ('ascmid', [0.5, 1, 2]),
#                 ('ascmax', [0, 0, 1])]  # ('descmid', [2, 1, 0.5]), ('descmax', [1, 0, 0]),
# TODO: Don't know what this is at all.
KS = [21]  # 14,
# TODO: use drmaa and a job template.
QSUB_STR = 'qsub -N {job_name} -P proj_covid -q all.q -l m_mem_free=6G -l fthread=3 -o omp_num_threads=3 '\
    '-e /share/temp/sgeoutput/covid_deaths '\
    '{code_dir}/tools/{env}_env.sh {code_dir}/src/covid_model_deaths/model.py '\
    '--model_location {model_location} --model_location_id {model_location_id} --data_file {data_file} '\
    '--cov_file {cov_file} --peaked_file {peaked_file} --output_dir {output_dir} --last_day_file {last_day_file} '\
    ' --covariate_effect {covariate_effect} --n_draws {n_draws}'
# QSUB_STR = 'qsub -N {job_name} -P proj_covid -q d.q -b y -l m_mem_free=6G -l fthread=3 '\
#            '-o /share/temp/sgeoutput/collijk/output/ '\
#            '-e /share/temp/sgeoutput/collijk/errors/ '\
#            '{python} {code_dir}/model.py '\
#            '--model_location {model_location} --model_location_id {model_location_id} --data_file {data_file} '\
#            '--cov_file {cov_file} --peaked_file {peaked_file} --output_dir {output_dir} --n_draws={n_draws}'
# FIXME: Defined in multiple places.
RATE_THRESHOLD = -15


def submit_curvefit(job_name: str, location_id: int, code_dir: str, env: str, model_location: str,
                    model_location_id: int, data_file: str, cov_file: str, last_day_file: str,
                    peaked_file: str, output_dir: str, covariate_effect: str, n_draws: int):
                    # model_location_id: int, data_file: str, cov_file: str, peaked_file: str, output_dir: str,
                    # n_draws: int, python: str, verbose: bool = False):
    qsub_str = QSUB_STR.format(
        job_name=job_name,
        location_id=location_id,
        code_dir=code_dir,
        env=env,
        python=python,
        # FIXME: Abstract string formatting somewhere else.
        model_location=sanitize(model_location),
        model_location_id=model_location_id,
        data_file=data_file.replace(' ', '\ ').replace('(', '\(').replace(')', '\)'),
        cov_file=cov_file.replace(' ', '\ ').replace('(', '\(').replace(')', '\)'),
        last_day_file=last_day_file.replace(' ', '\ ').replace('(', '\(').replace(')', '\)'),
        peaked_file=peaked_file.replace(' ', '\ ').replace('(', '\(').replace(')', '\)'),
        output_dir=output_dir.replace(' ', '\ ').replace('(', '\(').replace(')', '\)'),
        covariate_effect=covariate_effect,
        # data_file=sanitize(data_file),
        # cov_file=sanitize(cov_file),
        # peaked_file=sanitize(peaked_file),
        # output_dir=sanitize(output_dir),
        n_draws=n_draws
    )
    # if verbose:
    #     print(qsub_str)
    # job_str = ''
    # while not job_str:
    #     job_str = os.popen(qsub_str).read()
    #     if not job_str:
    #         print("Job submission failed. Retrying in 30 seconds...")
    #         print("Please try running qstat.")
    #         time.sleep(30)
    job_str = os.popen(qsub_str).read()
    print(job_str)


def sanitize(shell_string):
    shell_string = shell_string.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
    return f'"{shell_string}"'


def get_peak_date(past_draw_df, avg_df):
    draw_cols = [f'draw_{i}' for i in range(1000)]
    df_len = len(avg_df)

    # combine past and future model data
    future_draw_df = avg_df.loc[avg_df['observed'] == False].copy()
    future_draw_df['date'] = pd.to_datetime(future_draw_df['date'])
    if 'draw_999' not in past_draw_df.columns:
        past_draw_df['draw_999'] = past_draw_df['draw_998']
    if 'draw_999' not in future_draw_df.columns:
        future_draw_df['draw_999'] = future_draw_df['draw_998']
    peak_df = past_draw_df.append(future_draw_df[past_draw_df.columns])
    peak_df = peak_df.sort_values(['location_id', 'date']).reset_index(drop=True)
    n_locs = peak_df['location_id'].unique().size

    # get daily deaths
    daily = peak_df[draw_cols].values[1:] - peak_df[draw_cols].values[:-1]
    peak_df['start_date'] = peak_df.groupby('location_id', as_index=False)['date'].transform(min)
    peak_df = peak_df[1:]
    peak_df[draw_cols] = daily
    peak_df = peak_df.loc[peak_df['date'] != peak_df['start_date']]

    # take mean and find peak (taking first day at peak if multiplt)
    peak_df['mean_daily'] = peak_df[draw_cols].mean(axis=1)
    peak_df['peak'] = peak_df.groupby('location_id', as_index=False)['mean_daily'].transform(max)
    peak_df = peak_df.loc[peak_df['mean_daily'] == peak_df['peak']]
    if len(peak_df) < n_locs:
        raise ValueError('Not all locations in peak data')
    peak_df = peak_df.groupby('location_id', as_index=False)['date'].min()

    # attach data
    peak_df['peak_date'] = True
    df = avg_df.copy()
    df = peak_df[['location_id', 'date', 'peak_date']].merge(df, how='right')
    df['peak_date'] = df['peak_date'].fillna(False)
    if len(df) != df_len:
        raise ValueError('Have changed data dimensions somehow.')
    df = df.sort_values(['location_id', 'date']).reset_index(drop=True)

    return df[['location_id', 'location', 'date', 'peak_date', 'observed'] + draw_cols]


# FIXME: I think a lot of this is shared with stuff in compare_moving_average.py.
class CompareModelDeaths:
    # FIXME: mutable default arg.
    def __init__(self, old_draw_path: str, new_draw_path: str, draws: List[str] = [f'draw_{i}' for i in range(1000)]):
        self.old_df = pd.read_csv(old_draw_path)
        self.new_df = pd.read_csv(new_draw_path)
        self.draws = draws

    @staticmethod
    def _get_deaths_per_day(draw_df: pd.DataFrame, draws: List[str]) -> pd.DataFrame:
        draw_df = draw_df.sort_values(['location', 'date']).reset_index(drop=True).copy()
        delta = draw_df[draws].values[1:, :] - draw_df[draws].values[:-1, :]
        draw_df.iloc[1:][draws] = delta
        draw_df['day0'] = draw_df.groupby('location', as_index=False)['date'].transform('min')
        draw_df = draw_df.loc[draw_df['date'] != draw_df['day0']]
        draw_df['val_mean'] = draw_df[draws].mean(axis=1)
        draw_df['val_lower'] = np.percentile(draw_df[draws], 2.5, axis=1)
        draw_df['val_upper'] = np.percentile(draw_df[draws], 97.5, axis=1)

        return draw_df[['location', 'date', 'val_mean', 'val_lower', 'val_upper']]

    def _summarize_draws(self, agg_location: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                     pd.DataFrame, pd.DataFrame]:
        # old data
        old_df = self.old_df.copy()
        old_df['date'] = pd.to_datetime(old_df['date'])
        if agg_location is not None:
            agg_old_df = old_df.groupby('date', as_index=False)[self.draws].sum()
            agg_old_df['location'] = agg_location
            agg_old_df['location_id'] = -1
            agg_old_df['observed'] = False
            agg_old_df['peak_date'] = False
            old_df = agg_old_df[old_df.columns].append(old_df).reset_index(drop=True)
        old_df['val_mean'] = old_df[self.draws].mean(axis=1)
        old_df['val_lower'] = np.percentile(old_df[self.draws], 2.5, axis=1)
        old_df['val_upper'] = np.percentile(old_df[self.draws], 97.5, axis=1)
        old_daily_df = self._get_deaths_per_day(old_df, self.draws)
        old_df = old_df[['location', 'date', 'val_mean', 'val_lower', 'val_upper']]

        # new data
        new_df = self.new_df.copy()
        new_df['date'] = pd.to_datetime(new_df['date'])
        if agg_location is not None:
            agg_new_df = new_df.groupby('date', as_index=False)[self.draws].sum()
            agg_new_df['location'] = agg_location
            agg_new_df['location_id'] = -1
            agg_new_df['observed'] = False
            agg_new_df['peak_date'] = False
            new_df = agg_new_df[new_df.columns].append(new_df).reset_index(drop=True)
        new_df['val_mean'] = new_df[self.draws].mean(axis=1)
        new_df['val_lower'] = np.percentile(new_df[self.draws], 2.5, axis=1)
        new_df['val_upper'] = np.percentile(new_df[self.draws], 97.5, axis=1)
        new_daily_df = self._get_deaths_per_day(new_df, self.draws)
        new_df = new_df[['location', 'date', 'val_mean', 'val_lower', 'val_upper']]

        return old_df, old_daily_df, new_df, new_daily_df

    def make_some_pictures(self, pdf_out_path: str, agg_location: str = None) -> None:
        old_df, old_daily_df, new_df, new_daily_df = self._summarize_draws(agg_location)
        with PdfPages(pdf_out_path) as pdf:
            for location in new_df['location'].unique():
                fig, ax = plt.subplots(1, 2, figsize=(16.5, 8.5))
                # cumulative
                ax[0].fill_between(
                    old_df.loc[old_df['location'] == location, 'date'],
                    old_df.loc[old_df['location'] == location, 'val_lower'],
                    old_df.loc[old_df['location'] == location, 'val_upper'],
                    alpha=0.25, color='dodgerblue'
                )
                ax[0].plot(
                    old_df.loc[old_df['location'] == location, 'date'],
                    old_df.loc[old_df['location'] == location, 'val_mean'],
                    color='dodgerblue', label='old'
                )
                ax[0].fill_between(
                    new_df.loc[new_df['location'] == location, 'date'],
                    new_df.loc[new_df['location'] == location, 'val_lower'],
                    new_df.loc[new_df['location'] == location, 'val_upper'],
                    alpha=0.25, color='firebrick'
                )
                ax[0].plot(
                    new_df.loc[new_df['location'] == location, 'date'],
                    new_df.loc[new_df['location'] == location, 'val_mean'],
                    color='firebrick', label='new'
                )
                ax[0].set_xlabel('Date')
                ax[0].set_ylabel('Cumulative deaths')
                # plt.fill_between(
                #     alt_df.loc[alt_df['location'] == location, 'date'],
                #     alt_df.loc[alt_df['location'] == location, 'val_lower'],
                #     alt_df.loc[alt_df['location'] == location, 'val_upper'],
                #     alpha=0.25, color='forestgreen'
                # )
                # plt.plot(
                #     alt_df.loc[alt_df['location'] == location, 'date'],
                #     alt_df.loc[alt_df['location'] == location, 'val_mean'],
                #     color='forestgreen', label='new model/old data'
                # )

                # daily
                ax[1].fill_between(
                    old_daily_df.loc[old_daily_df['location'] == location, 'date'],
                    old_daily_df.loc[old_daily_df['location'] == location, 'val_lower'],
                    old_daily_df.loc[old_daily_df['location'] == location, 'val_upper'],
                    alpha=0.25, color='dodgerblue'
                )
                ax[1].plot(
                    old_daily_df.loc[old_daily_df['location'] == location, 'date'],
                    old_daily_df.loc[old_daily_df['location'] == location, 'val_mean'],
                    color='dodgerblue', label='old'
                )

                ax[1].fill_between(
                    new_daily_df.loc[new_daily_df['location'] == location, 'date'],
                    new_daily_df.loc[new_daily_df['location'] == location, 'val_lower'],
                    new_daily_df.loc[new_daily_df['location'] == location, 'val_upper'],
                    alpha=0.25, color='firebrick'
                )
                ax[1].plot(
                    new_daily_df.loc[new_daily_df['location'] == location, 'date'],
                    new_daily_df.loc[new_daily_df['location'] == location, 'val_mean'],
                    color='firebrick', label='new'
                )
                ax[1].set_xlabel('Date')
                ax[1].set_ylabel('Daily deaths')

                plt.xticks(rotation=60)
                plt.suptitle(location)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
