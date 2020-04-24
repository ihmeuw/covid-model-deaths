from datetime import datetime, timedelta
import functools
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

from db_queries import get_location_metadata
import dill as pickle
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import tqdm
import yaml

import covid_model_deaths
from covid_model_deaths.compare_moving_average import CompareAveragingModelDeaths
from covid_model_deaths.data import get_input_data, plot_crude_rates, DeathModelData, LeadingIndicator
from covid_model_deaths.drawer import Drawer
from covid_model_deaths.impute_death_threshold import impute_death_threshold as impute_death_threshold_
import covid_model_deaths.globals as cmd_globals
from covid_model_deaths.globals import COLUMNS, LOCATIONS
from covid_model_deaths.moving_average import moving_average_predictions
from covid_model_deaths.social_distancing_cov import SocialDistCov
from covid_model_deaths.utilities import submit_curvefit, CompareModelDeaths


class Checkpoint:

    def __init__(self, output_dir: str):
        self.checkpoint_dir = Path(output_dir) / 'checkpoint'
        self.checkpoint_dir.mkdir(mode=0o775, exist_ok=True)
        self.cache = {}

    def write(self, key, data):
        if key in self.cache:
            logger.warning(f"Overwriting {key} in checkpoint data.")
        self.cache[key] = data
        with (self.checkpoint_dir / f"{key}.pkl").open('wb') as key_file:
            pickle.dump(data, key_file, -1)

    def load(self, key):
        if key in self.cache:
            logger.info(f'Loading {key} from in memory cache.')
        elif (self.checkpoint_dir / f'{key}.pkl').exists():
            logger.info(f'Reading {key} from checkpoint dir {self.checkpoint_dir}.')
            with (self.checkpoint_dir / f"{key}.pkl").open('rb') as key_file:
                self.cache[key] = pickle.load(key_file)
        else:
            raise ValueError(f'No checkpoint data found for {key}')
        return self.cache[key]

    def __repr__(self):
        return f'Checkpoint({str(self.checkpoint_dir)})'


def run_us_model(input_data_version: str,
                 peak_file: str,
                 r0_file: str,
                 output_path: str,
                 datestamp_label: str,
                 r0_locs: List[int],
                 yesterday_draw_path: str,
                 before_yesterday_draw_path: str,
                 previous_average_draw_path: str) -> None:
    full_df = get_input_data('full_data', input_data_version)
    death_df = get_input_data('deaths', input_data_version)
    age_pop_df = get_input_data('age_pop', input_data_version)
    age_death_df = get_input_data('age_death', input_data_version)
    get_input_data('us_pops').to_csv(f'{output_path}/pops.csv', index=False)

    backcast_output_path = f'{output_path}/backcast_for_case_to_death.csv'
    threshold_dates_output_path = f'{output_path}/threshold_dates.csv'
    dcr_path = f'{output_path}/lagged_death_to_case_ratios.csv'
    dhr_path = f'{output_path}/lagged_death_to_hosp_ratios.csv'
    li_path = f'{output_path}/leading_indicator.csv'
    raw_draw_path = f'{output_path}/state_data.csv'
    model_type_path = f'{output_path}/state_models_used.csv'
    average_draw_path = f'{output_path}/past_avg_state_data.csv'

    plot_crude_rates(death_df, level='subnat')

    cases_and_backcast_deaths = make_cases_and_backcast_deaths(full_df, death_df, age_pop_df, age_death_df)
    cases_and_backcast_deaths.to_csv(backcast_output_path, index=False)

    in_us = cases_and_backcast_deaths[COLUMNS.country] == LOCATIONS.usa.name
    state_level = ~cases_and_backcast_deaths[COLUMNS.state].isnull()
    us_states = cases_and_backcast_deaths.loc[in_us & state_level, COLUMNS.state].unique().to_list()

    us_threshold_dates = impute_death_threshold(cases_and_backcast_deaths, location_list=us_states)
    us_threshold_dates.to_csv(threshold_dates_output_path, index=False)

    us_date_mean_df = make_date_mean_df(us_threshold_dates)

    last_day_df = make_last_day_df(full_df, us_date_mean_df)
    
    us_location_ids, us_location_names = get_us_location_ids_and_names(full_df)
    
    dcr_df, dhr_df, li_df = make_leading_indicator(full_df.loc[full_df[COLUMNS.location_id].isin(us_location_ids)])
    dcr_df.to_csv(dcr_path, index=False)
    dhr_df.to_csv(dhr_path, index=False)
    li_df.to_csv(li_path, index=False)
    li_df = li_df[[COLUMNS.location_id, COLUMNS.date, COLUMNS.ln_age_death_rate]]
    li_df = li_df.loc[~li_df[COLUMNS.ln_age_death_rate].isnull()]

    submodel_dict = submit_models(full_df, death_df, age_pop_df, age_death_df, us_date_mean_df, li_df,
                                  us_location_ids, us_location_names, r0_locs,
                                  peak_file, output_path, input_data_version, r0_file,
                                  str(Path(covid_model_deaths.__file__).parent))

    in_us = full_df[COLUMNS.country] == LOCATIONS.usa.name
    state_level = ~full_df[COLUMNS.state].isnull()
    usa_obs_df = full_df[in_us & state_level]

    draw_dfs, past_draw_dfs, models_used, days, ensemble_draws_dfs = compile_draws(us_location_ids,
                                                                                   us_location_names,
                                                                                   submodel_dict,
                                                                                   usa_obs_df,
                                                                                   us_threshold_dates,
                                                                                   age_pop_df)
    if 'location' not in models_used:
        raise ValueError('No location-specific draws used, must be using wrong tag')

    draw_df = pd.concat(draw_dfs)
    model_type_df = pd.DataFrame({
        'location': us_location_names,
        'model_used': models_used
    })
    draw_df.to_csv(raw_draw_path, index=False)
    model_type_df.to_csv(model_type_path, index=False)

    ensemble_path = make_and_save_draw_plots(output_path, us_location_ids, us_location_names,
                                             ensemble_draws_dfs, days, models_used)

    average_draw_df = average_draws(output_path, yesterday_draw_path, before_yesterday_draw_path)
    average_draw_df.to_csv(average_draw_path)

    moving_average_path = make_and_save_compare_average_plots(output_path, raw_draw_path, average_draw_path,
                                                              yesterday_draw_path, before_yesterday_draw_path)

    compare_to_previous_path = make_and_save_compare_to_previous_plots(output_path, average_draw_path,
                                                                       previous_average_draw_path)

    send_plots_to_diagnostics(datestamp_label, ensemble_path, moving_average_path, compare_to_previous_path)


def make_cases_and_backcast_deaths(full_df: pd.DataFrame, death_df: pd.DataFrame,
                                   age_pop_df: pd.DataFrame, age_death_df: pd.DataFrame,
                                   location_ids: List[int], subnat: bool=True) -> pd.DataFrame:
    backcast_deaths_df = backcast_deaths_parallel(location_ids, death_df, age_pop_df, age_death_df, subnat)

    full_df_columns = [COLUMNS.location_id, COLUMNS.state, COLUMNS.country, COLUMNS.date,
                       COLUMNS.confirmed, COLUMNS.confirmed_case_rate]
    cases_and_backcast_deaths = full_df[full_df_columns].merge(backcast_deaths_df, how='outer').reset_index(drop=True)

    country_level = cases_and_backcast_deaths[COLUMNS.state].isnull()
    cases_and_backcast_deaths.loc[country_level, COLUMNS.state] = cases_and_backcast_deaths[COLUMNS.country]
    cases_and_backcast_deaths[COLUMNS.location_id] = cases_and_backcast_deaths[COLUMNS.location_id].astype(int)

    return cases_and_backcast_deaths


def impute_death_threshold(cases_and_backcast_deaths_df: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
    threshold_dates = impute_death_threshold_(cases_and_backcast_deaths_df,
                                              location_list=loc_df[COLUMNS.location_id].unique().tolist(),
                                              ln_death_rate_threshold=cmd_globals.LN_MORTALITY_RATE_THRESHOLD)
    loc_df = (loc_df
              .loc[:, [COLUMNS.location, COLUMNS.location_id]]
              .rename(columns={COLUMNS.location: COLUMNS.location_bad}))
    threshold_dates = loc_df.merge(threshold_dates)
    return threshold_dates


def make_date_mean_df(threshold_dates: pd.DataFrame) -> pd.DataFrame:
    # get mean data from dataset
    date_draws = [i for i in threshold_dates.columns if i.startswith('death_date_draw_')]
    date_mean_df = threshold_dates.copy()
    date_mean_df[COLUMNS.threshold_date] = date_mean_df.apply(
        lambda x: datetime.strptime(date_mean(x[date_draws]).strftime('%Y-%m-%d'), '%Y-%m-%d'),
        axis=1
    )
    date_mean_df[COLUMNS.country] = LOCATIONS.usa.name
    date_mean_df = date_mean_df.rename(index=str, columns={COLUMNS.location_bad: COLUMNS.location})
    date_mean_df = date_mean_df[[COLUMNS.location_id, COLUMNS.location, COLUMNS.country, COLUMNS.threshold_date]]
    return date_mean_df


def make_last_day_df(full_df: pd.DataFrame, date_mean_df: pd.DataFrame) -> pd.DataFrame:
    # prepare last day dataset
    last_day_df = full_df.copy()
    last_day_df[COLUMNS.last_day] = (last_day_df
                                     .groupby(COLUMNS.location_id, as_index=False)[COLUMNS.date]
                                     .transform(max))
    last_day_df = last_day_df.loc[last_day_df[COLUMNS.date] == last_day_df[COLUMNS.last_day]].reset_index(drop=True)
    
    last_day_df[COLUMNS.location_id] = last_day_df[COLUMNS.location_id].astype(int)
    # TODO: Document whatever is happening here.
    last_day_df.loc[last_day_df[COLUMNS.death_rate] == 0, COLUMNS.death_rate] = 0.1 / last_day_df[COLUMNS.population]
    last_day_df[COLUMNS.ln_death_rate] = np.log(last_day_df[COLUMNS.death_rate])
    last_day_df = last_day_df[[COLUMNS.location_id, COLUMNS.ln_death_rate, COLUMNS.date]].merge(date_mean_df)
    last_day_df[COLUMNS.days] = (last_day_df[COLUMNS.date] - last_day_df[COLUMNS.threshold_date])
    last_day_df[COLUMNS.days] = last_day_df[COLUMNS.days].apply(lambda x: x.days)
    last_day_df = last_day_df.loc[last_day_df[COLUMNS.days] > 0]
    return last_day_df[[COLUMNS.location_id, COLUMNS.ln_death_rate, COLUMNS.days]]


def make_leading_indicator(full_df: pd.DataFrame) -> pd.DataFrame:
    leading = LeadingIndicator(full_df)
    dcr_df, dhr_df, li_df = leading.produce_deaths()
    return dcr_df, dhr_df, li_df


def submit_models(full_df: pd.DataFrame, death_df: pd.DataFrame, age_pop_df: pd.DataFrame,
                  age_death_df: pd.DataFrame, date_mean_df: pd.DataFrame, li_df: pd.DataFrame,
                  loc_df: pd.DataFrame, r0_locs: List[int], peak_file: str, output_directory: str, 
                  data_version: str, r0_file: str, code_dir: str, verbose: bool = False) -> Dict:
    submodel_dict = {}
    N = len(loc_df)
    i = 0
    nursing_home_locations = [LOCATIONS.life_care.name]
    for _, (location_id, location_name) in tqdm.tqdm(loc_df[[COLUMNS.location_id, COLUMNS.location]].iterrows(), total=len(loc_df)):
        location_id = int(location_id)
        i += 1
        level = loc_df.set_index(COLUMNS.location_id).at[location_id, COLUMNS.level]
        subnat = not level == 0
        mod = DeathModelData(death_df, age_pop_df, age_death_df, location_id, 'threshold',
                             subnat=subnat, rate_threshold=cmd_globals.LN_MORTALITY_RATE_THRESHOLD)
        if location_name in nursing_home_locations:
            # save only nursing homes
            mod_df = mod.df.copy()
        else:
            # save only others
            mod_df = mod.df.loc[~mod.df[COLUMNS.location].isin(nursing_home_locations)].reset_index(drop=True)
        mod_df = mod_df.loc[~(mod_df[COLUMNS.deaths].isnull())].reset_index(drop=True)

        # flag as true data
        mod_df[COLUMNS.pseudo] = 0

        # tack on deaths from cases if in dataset
        # South Dakota, Iowa
        if location_id in full_df[COLUMNS.location_id].tolist() and location_id not in [564, 538]:
            # get future days
            last_date = full_df.loc[full_df[COLUMNS.location_id] == location_id, COLUMNS.date].max()
            loc_cd_df = li_df.loc[(li_df[COLUMNS.location_id] == location_id)
                                  & (li_df[COLUMNS.date] > last_date)].reset_index(drop=True)
            loc_cd_df[COLUMNS.population] = full_df.loc[full_df[COLUMNS.location_id] == location_id,
                                                        COLUMNS.population].max()  # all the same...
            loc_cd_df[COLUMNS.pseudo] = 1

            if not loc_cd_df.empty:
                # convert to days
                if location_id in mod_df[COLUMNS.location_id].tolist():
                    last_day = mod_df.loc[mod_df[COLUMNS.location_id] == location_id, COLUMNS.days].max()
                    loc_cd_df[COLUMNS.days] = last_day + 1 + loc_cd_df.index
                else:
                    threshold = date_mean_df.loc[date_mean_df[COLUMNS.location_id] == location_id,
                                                 COLUMNS.threshold_date].item()
                    loc_cd_df[COLUMNS.days] = loc_cd_df[COLUMNS.date].apply(lambda x: (x - threshold).days)
                loc_cd_df = loc_cd_df.loc[loc_cd_df[COLUMNS.days] >= 0]

                # stick on to dataset
                mod_df = mod_df.append(loc_cd_df)
                mod_df = mod_df.sort_values([COLUMNS.location_id, COLUMNS.days]).reset_index(drop=True)

        # figure out which models we are running (will need to check about R0=1 model)
        submodels = cmd_globals.MOBILITY_SOURCES.copy()
        if location_id in r0_locs:
            submodels += ['R0_35', 'R0_50', 'R0_65']
        submodel_dirs = setup_submodel_dirs(output_directory, cmd_globals.MOBILITY_SOURCES)

        n_draws_list = get_draw_list(n_scenarios=len(submodel_dirs))

        # store this information
        submodel_dict.update({
            int(location_id): {
                'submodel_dirs': submodel_dirs,
                'n_draws_list': n_draws_list
            }
        })

        n_i = 0
        for cov_source in submodels:
            if cov_source in cmd_globals.MOBILITY_SOURCES:
                covariate_effect = 'gamma'
            else:
                covariate_effect = 'beta'
            for k in cmd_globals.KS:
                # drop back-cast for modeling file, but NOT for the social distancing covariate step
                model_out_dir = f'{output_directory}/model_data_{cov_source}_{k}'
                mod_df.to_csv(f'{model_out_dir}/{location_id}.csv', index=False)
                sd_cov = SocialDistCov(mod_df, date_mean_df, data_version=data_version)
                if cov_source in cmd_globals.MOBILITY_SOURCES:
                    sd_cov_df = sd_cov.get_cov_df(weights=[None], k=k, empirical_weight_source=cov_source)
                else:
                    sd_cov_df = sd_cov.get_cov_df(weights=[None], k=k, empirical_weight_source=cov_source,
                                                  R0_file=r0_file)
                sd_cov_df.to_csv(f'{model_out_dir}/{location_id}_covariate.csv', index=False)
                if not os.path.exists(f'{model_out_dir}/{location_id}'):
                    os.mkdir(f'{model_out_dir}/{location_id}')

                submit_curvefit(job_name=f'curve_model_{location_id}_{cov_source}_{k}',
                                location_id=location_id,
                                model_file=f'{code_dir}/model.py',
                                model_location_id=location_id,
                                data_file=f'{model_out_dir}/{location_id}.csv',
                                cov_file=f'{model_out_dir}/{location_id}_covariate.csv',
                                last_day_file=f'{output_directory}/last_day.csv',
                                peaked_file=peak_file,
                                output_dir=f'{model_out_dir}/{location_id}',
                                covariate_effect=covariate_effect,
                                n_draws=n_draws_list[n_i],
                                python=shutil.which('python'),
                                verbose=verbose)
                n_i += 1
    return submodel_dict


def compile_draws(loc_df: pd.DataFrame, submodel_dict: Dict,
                  obs_df: pd.DataFrame, threshold_dates: pd.DataFrame, age_pop_df: pd.DataFrame
                  ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List, List, List[pd.DataFrame]]:
    draw_dfs = []
    past_draw_dfs = []
    models_used = []
    days_ = []
    ensemble_draws_dfs = []
    for _, (location_id, location_name) in tqdm.tqdm(loc_df[[COLUMNS.location_id, COLUMNS.location]].iterrows(), total=len(loc_df)):
        # # identify peak duration
        # if int(location_id) in peak_dur_df['location_id'].to_list():
        #     print(f'{location_name}: observed peak')
        #     peak_duration = peak_dur_df.loc[peak_dur_df['location_id'] == int(location_id), 'peak durations'].item()
        # else:
        #     print(f'{location_name}: average peak')
        #     peak_duration = 5
        # peak_duration = int(np.round(peak_duration))
        # print(f'Peak length: {peak_duration}')
        peak_duration = 1
        # get draws
        data_draws = Drawer(
            ensemble_dirs=submodel_dict[int(location_id)]['submodel_dirs'],
            n_draws_list=submodel_dict[int(location_id)]['n_draws_list'],
            location_name=location_name,
            location_id=int(location_id),
            peak_duration=peak_duration,
            obs_df=obs_df.loc[obs_df[COLUMNS.location_id] == location_id],
            date_draws=threshold_dates.loc[threshold_dates[COLUMNS.location_bad] == location_name,
                                           [i for i in threshold_dates.columns if i.startswith('death_date_draw_')]].values,
            population=age_pop_df.loc[age_pop_df[COLUMNS.location_id] == int(location_id), COLUMNS.population].sum()
        )
        try:
            draw_df, past_draw_df, model_used, days, ensemble_draws = data_draws.get_dated_draws()
        except Exception as e:
            print(e)
            print('No draws for ', location_name, location_id)
            continue
        draw_dfs.append(draw_df)
        past_draw_dfs.append(past_draw_df)
        models_used.append(model_used)
        days_.append(days)
        ensemble_draws_dfs.append(ensemble_draws)
    return draw_dfs, past_draw_dfs, models_used, days_, ensemble_draws_dfs


def average_draws(raw_draw_path: str,
                  yesterday_path: str, before_yesterday_path: str) -> pd.DataFrame:
    avg_df = moving_average_predictions(
        today_data_path=raw_draw_path,
        yesterday_data_path=yesterday_path,
        day_before_yesterday_path=before_yesterday_path
    )
    avg_df['date'] = pd.to_datetime(avg_df['date'])
    return avg_df


def make_and_save_draw_plots(output_dir: str, loc_df: pd.DataFrame,
                             ensemble_draw_dfs: List[Dict[str, pd.DataFrame]], days_: List, models_used: List,
                             age_pop_df: pd.DataFrame) -> str:
    # plot ensemble
    # ensemble plot settings
    color_dict = {
        'safegraph': 'dodgerblue',
        'google': 'forestgreen',
        'descartes': 'firebrick',
        # 'R0_35':'gold',
        # 'R0_50':'darkgrey',
        # 'R0_65':'darkviolet'
    }
    line_dict = {
        '21': '--'
    }
    # HACK: Pulled out of a notebook that relied on this only having one item.
    k = cmd_globals.KS[0]
    plot_vars = list(zip(loc_df[COLUMNS.location_id], loc_df[COLUMNS.location], ensemble_draw_dfs, days_, models_used))
    plot_path = f'{output_dir}/ensemble_plot.pdf'
    with PdfPages(plot_path) as pdf:
        for location_id, location_name, ensemble_draws, days, model_used in tqdm.tqdm(plot_vars, total=len(plot_vars)):
            fig, ax = plt.subplots(1, 2, figsize=(11, 8.5))
            for label, draws in ensemble_draws.items():
                label = label.split('model_data_')[1]
                draws = np.exp(draws) * age_pop_df.loc[age_pop_df[COLUMNS.location_id] == int(location_id), COLUMNS.population].sum()
                deaths_mean = draws.mean(axis=0)
                deaths_lower = np.percentile(draws, 2.5, axis=0)
                deaths_upper = np.percentile(draws, 97.5, axis=0)

                d_deaths_mean = (draws[:, 1:] - draws[:, :-1]).mean(axis=0)
                d_deaths_lower = np.percentile(draws[:, 1:] - draws[:, :-1], 2.5, axis=0)
                d_deaths_upper = np.percentile(draws[:, 1:] - draws[:, :-1], 97.5, axis=0)

                # cumulative
                ax[0].fill_between(days,
                                   deaths_lower, deaths_upper,
                                   color=color_dict['_'.join(label.split('_')[:-1])],
                                   linestyle=line_dict[label.split('_')[-1]],
                                   alpha=0.25)
                ax[0].plot(days, deaths_mean,
                           c=color_dict['_'.join(label.split('_')[:-1])],
                           linestyle=line_dict[label.split('_')[-1]])
                ax[0].set_xlabel('Date')
                ax[0].set_ylabel('Cumulative death rate')

                # daily
                ax[1].fill_between(days[1:],
                                   d_deaths_lower, d_deaths_upper,
                                   color=color_dict['_'.join(label.split('_')[:-1])],
                                   linestyle=line_dict[label.split('_')[-1]],
                                   alpha=0.25)
                ax[1].plot(days[1:], d_deaths_mean,
                           c=color_dict['_'.join(label.split('_')[:-1])],
                           linestyle=line_dict[label.split('_')[-1]],
                           label=label.replace('model_data_', ''))
                ax[1].set_xlabel('Date')
                ax[1].set_ylabel('Daily death rates')

            ax[1].legend(loc=2)
            plt.suptitle(f'{location_name} ({model_used})')
            plt.tight_layout()
            pdf.savefig()
    return plot_path


def make_and_save_compare_average_plots(output_dir: str, raw_draw_path: str, average_draw_path: str,
                                        yesterday_draw_path: str, before_yesterday_draw_path: str, label: str) -> str:
    plotter = CompareAveragingModelDeaths(
        raw_draw_path=raw_draw_path,
        average_draw_path=average_draw_path,
        yesterday_draw_path=yesterday_draw_path,
        before_yesterday_draw_path=before_yesterday_draw_path
    )
    plot_path = f'{output_dir}/moving_average_compare.pdf'
    plotter.make_some_pictures(plot_path, label)
    return plot_path


def make_and_save_compare_to_previous_plots(output_dir: str, today_average_path: str,
                                            previous_average_path: str, label: str) -> str:
    plotter = CompareModelDeaths(
        old_draw_path=previous_average_path,
        new_draw_path=today_average_path
    )
    plot_path = f'{output_dir}/compare_to_previous.pdf'
    plotter.make_some_pictures(plot_path, label)
    return plot_path


def send_plots_to_diagnostics(datestamp_label: str, *plot_paths: str) -> str:
    viz_dir = Path(f'/home/j/Project/covid/results/diagnostics/deaths/{datestamp_label}/')
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)
    for plot_path in plot_paths:
        plot_path = Path(plot_path)
        shutil.copyfile(src=plot_path, dst=viz_dir / plot_path.name)
    return str(viz_dir)


def get_backcast_location_ids(data: pd.DataFrame, most_detailed: bool = True) -> List[int]:
    rate_above_threshold = np.log(data[COLUMNS.death_rate]) > cmd_globals.LN_MORTALITY_RATE_THRESHOLD
    if most_detailed:
        in_location_set = ~data[COLUMNS.state].isnull()
    else:
        in_location_set = pd.Series(True, index=data.index)
    location_ids = sorted(data.loc[rate_above_threshold & in_location_set, COLUMNS.location_id].unique().tolist())
    return location_ids


def get_us_location_ids_and_names(full_df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    loc_df = get_location_metadata(location_set_id=cmd_globals.GBD_REPORTING_LOCATION_SET_ID,
                                   gbd_round_id=cmd_globals.GBD_2017_ROUND_ID)
    us_states = loc_df[COLUMNS.parent_id] == LOCATIONS.usa.id
    not_wa_state = loc_df[COLUMNS.location_name] != LOCATIONS.washington.name
    us_location_ids_except_wa = loc_df.loc[us_states & not_wa_state, COLUMNS.location_id].to_list()
    us_location_names_except_wa = loc_df.loc[us_states & not_wa_state, COLUMNS.location_name].to_list()

    wa_state_ids = []
    wa_state_names = []
    for wa_location in [LOCATIONS.other_wa_counties.name, LOCATIONS.king_and_snohomish.name, LOCATIONS.life_care.name]:
        wa_state_ids += [int(full_df.loc[full_df[COLUMNS.state] == wa_location, COLUMNS.location_id].unique().item())]
        wa_state_names += [wa_location]

    us_location_ids = us_location_ids_except_wa + wa_state_ids
    us_location_names = us_location_names_except_wa + wa_state_names
    return us_location_ids, us_location_names


def backcast_deaths_parallel(location_ids: List[int], death_df: pd.DataFrame,
                             age_pop_df: pd.DataFrame, age_death: pd.DataFrame, subnat: bool) -> pd.DataFrame:
    _combiner = functools.partial(backcast_deaths,
                                  death_df=death_df,
                                  age_pop_df=age_pop_df,
                                  age_death_df=age_death,
                                  subnat=subnat)
    with multiprocessing.Pool(20) as p:
        backcast_deaths_dfs = list(tqdm.tqdm(p.imap(_combiner, location_ids), total=len(location_ids)))
    return pd.concat(backcast_deaths_dfs)


def backcast_deaths(location_id: int, death_df: pd.DataFrame,
                    age_pop_df: pd.DataFrame, age_death_df: pd.DataFrame, subnat: bool) -> pd.DataFrame:
    output_columns = [COLUMNS.location_id, COLUMNS.state, COLUMNS.country, COLUMNS.date,
                      COLUMNS.deaths, COLUMNS.death_rate, COLUMNS.population]
    death_model = DeathModelData(death_df,
                                 age_pop_df,
                                 age_death_df,
                                 location_id,
                                 'threshold',
                                 subnat=subnat,
                                 rate_threshold=cmd_globals.LN_MORTALITY_RATE_THRESHOLD)
    mod_df = death_model.df
    mod_df = mod_df.loc[mod_df[COLUMNS.location_id] == location_id].reset_index(drop=True)
    if len(mod_df) > 0:
        date0 = mod_df[COLUMNS.date].min()
        day0 = mod_df.loc[~mod_df[COLUMNS.date].isnull(), COLUMNS.days].min()
        mod_df.loc[mod_df[COLUMNS.days] == 0, COLUMNS.date] = date0 - timedelta(days=np.round(day0))
        mod_df = mod_df.loc[~((mod_df[COLUMNS.deaths].isnull()) & (mod_df[COLUMNS.date] == date0))]
        mod_df = mod_df.loc[~mod_df[COLUMNS.date].isnull()]
        mod_df.loc[mod_df[COLUMNS.death_rate].isnull(), COLUMNS.death_rate] = np.exp(mod_df[COLUMNS.ln_age_death_rate])
        mod_df.loc[mod_df[COLUMNS.deaths].isnull(), COLUMNS.deaths] = mod_df[COLUMNS.death_rate] * mod_df[COLUMNS.population]
        mod_df = mod_df.rename(index=str, columns={COLUMNS.location: COLUMNS.state})
    else:
        mod_df = pd.DataFrame(columns=output_columns)

    return mod_df[output_columns].reset_index(drop=True)


def date_mean(dates: pd.Series) -> datetime:
    dt_min = dates.min()
    deltas = pd.Series([x-dt_min for x in dates])
    return dt_min + deltas.sum() / len(deltas)


def get_draw_list(n_scenarios):
    n_draws_list = [int(1000 / n_scenarios)] * n_scenarios
    n_draws_list[-1] = n_draws_list[-1] + 1000 - np.sum(n_draws_list)

    return n_draws_list


def setup_submodel_dirs(output_directory: str, model_labels: List[str]) -> List[str]:
    model_out_dirs = []
    for model_label in model_labels:
        for k in cmd_globals.KS:
            # set up dirs
            model_out_dir = Path(f'{output_directory}/model_data_{model_label}_{k}')
            if not model_out_dir.exists():
                model_out_dir.mkdir(mode=0o775, exist_ok=True)
            model_out_dirs.append(str(model_out_dir))
    return model_out_dirs


def display_total_deaths(draw_df: pd.DataFrame):
    draw_cols = [f'draw_{i}' for i in range(1000)]
    nat_df = draw_df.groupby('date', as_index=False)[draw_cols].sum()
    nat_df = nat_df.loc[nat_df['date'] == pd.Timestamp('2020-07-15')]
    deaths_mean = int(nat_df[draw_cols].mean(axis=1).item())
    deaths_lower = int(np.percentile(nat_df[draw_cols], 2.5, axis=1).item())
    deaths_upper = int(np.percentile(nat_df[draw_cols], 97.5, axis=1).item())
    print(f'{deaths_mean:,} ({deaths_lower:,} - {deaths_upper:,})')
