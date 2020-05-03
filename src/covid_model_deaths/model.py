import argparse
from copy import deepcopy
import hashlib
import warnings

from curvefit.pipelines.flat_asymmetric_model import APFlatAsymmetricModel
from curvefit.pipelines.ap_model import APModel
from curvefit.core.functions import ln_gaussian_cdf, ln_gaussian_pdf
from curvefit.core.utils import truncate_draws, convex_combination, process_input
import dill as pickle
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from datetime import timedelta


warnings.filterwarnings('ignore')

RATE_THRESHOLD = -15  # should pass this in as argument
COVARIATE = 'cov_3w'
DATA_THRESHOLD = 18
PSEUDO_SE = 3
N_B = 29
PRED_DAYS = 150


def get_hash(key: str) -> int:
    """Gets a hash of the provided key.
    Parameters
    ----------
    key :
        A string used to create a seed for the random number generator.
    Returns
    -------
    int
        A hash of the provided key.
    """
    # 4294967295 == 2**32 - 1 which is the maximum allowable seed for a `numpy.random.RandomState`.
    return int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295


def ap_model(df, model_location, location_cov, n_draws,
             peaked_groups, exclude_groups, fix_gamma, fix_point, fix_day,
             pred_days=PRED_DAYS):
    # our dataset (rename days as model assumes it's lower case)
    df = df.copy()
    df = df.rename(index=str, columns={'Days':'days'})

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SET UP
    # basic information and model setting
    basic_info_dict = dict(
        all_cov_names=[COVARIATE],
        col_t='days',
        col_group='location',
        predict_space=ln_gaussian_pdf,
        col_obs_compare='ln asddr',
        peaked_groups=peaked_groups
    )
    basic_model_dict = dict(
        param_names=['alpha', 'beta', 'p'],
        col_covs=[['intercept'], [COVARIATE], ['intercept']],
        link_fun=[np.exp, lambda x: x, np.exp],
        var_link_fun=[lambda x: x, lambda x: x, lambda x: x]
    )

    # basic fit parameter
    dummy_gprior = [0.0, np.inf]
    dummy_uprior = [-np.inf, np.inf]
    zero_uprior = [0.0, 0.0]
    fe_init = np.array([-2.5, 28.0, -8.05])
    initial_scalars = [[-0.5, 0.0, 0.5], [0.0], [0.0]]
    fe_bounds = [[-np.inf, 0.0], [15.0, 100.0], [-10, -6]]
    options = {
        'ftol': 1e-10,
        'gtol': 1e-10,
        'maxiter': 500,
        'disp': False
    }
    basic_fit_dict = dict(
        fe_init=fe_init,
        fe_bounds=fe_bounds,
        re_bounds=[zero_uprior]*3,
        fe_gprior=[dummy_gprior]*3,
        re_gprior=[dummy_gprior]*3,
        options=options
    )
    basic_joint_model_fit_dict = dict(
        fe_gprior=[dummy_gprior]*3,
        re_bounds=[dummy_uprior]*3,
        re_gprior=[dummy_gprior, [0.0, 10.0], dummy_gprior],
        smart_initialize=True,
        smart_init_options=options,
        options={
            'ftol': 1e-10,
            'gtol': 1e-10,
            'maxiter': 10,
            'disp': False
        }
    )

    # draw related parameters
    draw_dict = dict(
        n_draws=n_draws,
        prediction_times = np.arange(pred_days),
        cv_lower_threshold=1e-4,
        cv_upper_threshold=1.,
        smoothed_radius=[5, 5],
        exclude_groups=exclude_groups,
        exclude_below=0,
        num_smooths=2
    )

    # for the convex combination
    start_day = 2
    end_day = 25

    # for prediction of places with no data
    alpha_times_beta = np.exp(0.7)
    obs_bounds = [40, np.inf] # filter the data rich models
    predict_cov = np.array([1.0, location_cov, 1.0]) # new covariates for the places.

    # tight prior control panel
    tight_info_dict = {
        **deepcopy(basic_info_dict),
        'fun': ln_gaussian_cdf,
        'col_obs': 'ln ascdr',
        'col_obs_se':'obs_se_tight',
        #'obs_se_func': lambda x: (1. / (1. + x)),
        'obs_se_func': None,
        'prior_modifier': lambda x: 10**(min(0.0, max(-1.0,
                    0.1*x - 1.5
        ))) / 10
    }
    tight_fit_dict = {
        **deepcopy(basic_fit_dict),
        'fun_gprior': [lambda params: params[0] * params[1], [np.exp(0.7), 1.0]]
    }

    # loose prior control panel
    loose_info_dict = {
        **deepcopy(basic_info_dict),
        'fun': ln_gaussian_cdf,
        'col_obs': 'ln ascdr',
        'col_obs_se':'obs_se_loose',
        #'obs_se_func': lambda x: (1 / (0.1 + x**1.4)),
        'obs_se_func': None,
        'prior_modifier': lambda x: 0.2
    }
    loose_fit_dict = {
        **deepcopy(basic_fit_dict),
        'fun_gprior': [lambda params: params[0] * params[1], dummy_gprior]
    }

    # prepare data (must exponentiate smoothed column, non-logged col is not smoothed)
    df['obs_se_tight'] = 1 / (1 + df['days'])
    df['obs_se_loose'] = 1 / (1 + df['days']**1.4)
    df.loc[df['pseudo'] == 1, 'obs_se_tight'] = PSEUDO_SE
    df.loc[df['pseudo'] == 1, 'obs_se_loose'] = PSEUDO_SE
    df['Age-standardized death rate'] = np.exp(df['ln(age-standardized death rate)'])
    df = process_input(df, 'location_id', 'days', 'Age-standardized death rate',
                       col_covs=[COVARIATE, 'intercept', 'obs_se_tight', 'obs_se_loose'])

    #############
    # RUN MODEL #
    #############
    # set up last info
    if fix_point is not None:
        last_info = {model_location:[fix_day, fix_point]}
    else:
        last_info = None

    # The Alpha Prior Model
    tight_model = APModel(
        all_data=df,
        **tight_info_dict,
        joint_model_fit_dict=basic_joint_model_fit_dict,
        basic_model_dict=basic_model_dict,
        fit_dict=tight_fit_dict
    )
    if fix_gamma:
        fe_bounds = tight_model.fit_dict['fe_bounds']
        tight_model.fit_dict.update({
            'fe_bounds': [fe_bounds[0], [1, 1], fe_bounds[2]]
        })
    tight_model.run(last_info=last_info,
                    initial_scalars=initial_scalars,
                    **draw_dict)
    loose_model = APModel(
        all_data=df,
        **loose_info_dict,
        joint_model_fit_dict=basic_joint_model_fit_dict,
        basic_model_dict=basic_model_dict,
        fit_dict=loose_fit_dict
    )
    if fix_gamma:
        fe_bounds = loose_model.fit_dict['fe_bounds']
        loose_model.fit_dict.update({
            'fe_bounds': [fe_bounds[0], [1, 1], fe_bounds[2]]
        })
    loose_model.run(last_info=last_info,
                    initial_scalars=initial_scalars,
                    **draw_dict)

    # get truncated draws
    tight_draws = tight_model.process_draws(draw_dict['prediction_times'],
                                            last_info=last_info)
    loose_draws = loose_model.process_draws(draw_dict['prediction_times'],
                                            last_info=last_info)
    combined_draws = {}
    for group in tight_draws.keys():
        draws = convex_combination(np.arange(tight_draws[group][1].shape[1]),
                                   tight_draws[group][1][np.argsort(tight_draws[group][1][:, -1]), :],
                                   loose_draws[group][1][np.argsort(loose_draws[group][1][:, -1]), :],
                                   basic_info_dict['predict_space'],
                                   start_day=start_day,
                                   end_day=end_day)
        if group == model_location and fix_point is not None:
            last_obs = fix_point
        else:
            last_obs = tight_model.models[group].obs[-1]
        combined_draws.update({
            group: (tight_draws[group][0],
                    np.log(np.exp(last_obs) + np.exp(draws).cumsum(axis=1)))
        })

    # get overall draws (using obs with > 1 data point)
    filtered_tight_models = tight_model.run_filtered_models(
        df=tight_model.all_data, obs_bounds=obs_bounds
    )
    overall_tight_draws = tight_model.create_overall_draws(
        draw_dict['prediction_times'], filtered_tight_models, predict_cov, alpha_times_beta=alpha_times_beta,
        sample_size=draw_dict['n_draws'], slope_at=10, epsilon=draw_dict['cv_lower_threshold']
    )
    filtered_loose_models = loose_model.run_filtered_models(
        df=loose_model.all_data, obs_bounds=obs_bounds
    )
    overall_loose_draws = loose_model.create_overall_draws(
        draw_dict['prediction_times'], filtered_loose_models, predict_cov, alpha_times_beta=alpha_times_beta,
        sample_size=draw_dict['n_draws'], slope_at=10, epsilon=draw_dict['cv_lower_threshold']
    )

    # get specs and truncate overall, then combine
    if model_location in list(combined_draws.keys()):
        # last_day = tight_model.models[model_location].t[-1]
        if fix_day is None:
            last_day = tight_model.models[model_location].t[-1]
        else:
            last_day = fix_day
        if fix_point is not None:
            last_obs = fix_point
        else:
            last_obs = tight_model.models[model_location].obs[-1]
        overall_time = draw_dict['prediction_times'][int(np.round(last_day)):]
    else:
        if fix_day is None:
            last_day = draw_dict['prediction_times'][0]
        else:
            last_day = fix_day
        if fix_point is not None:
            last_obs = fix_point
        else:
            last_obs = RATE_THRESHOLD
        overall_time = np.arange(last_day, pred_days)
    overall_tight_draws = truncate_draws(
        t=draw_dict['prediction_times'], draws=overall_tight_draws,
        draw_space=basic_info_dict['predict_space'],
        last_day=last_day,
        last_obs=last_obs,
        last_obs_space=tight_info_dict['fun']
    )
    overall_loose_draws = truncate_draws(
        t=draw_dict['prediction_times'], draws=overall_loose_draws,
        draw_space=basic_info_dict['predict_space'],
        last_day=last_day,
        last_obs=last_obs,
        last_obs_space=loose_info_dict['fun']
    )
    draws = convex_combination(np.arange(overall_tight_draws.shape[1]),
                               overall_tight_draws[np.argsort(overall_tight_draws[:, -1]), :],
                               overall_loose_draws[np.argsort(overall_loose_draws[:, -1]), :],
                               basic_info_dict['predict_space'],
                               start_day=start_day,
                               end_day=end_day)
    combined_draws.update({
        'overall': (overall_time[1:],
                    np.log(np.exp(last_obs) + np.exp(draws).cumsum(axis=1)))
    })

    return tight_model, loose_model, combined_draws


def ap_flat_asym_model(df, model_location, n_draws, peaked_groups, exclude_groups,
                       fix_point, fix_day, pred_days=PRED_DAYS):
    # our dataset (rename days as model assumes it's lower case)
    df = df.copy()
    df = df.rename(index=str, columns={'Days':'days'})

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SET UP
    # basic information and model setting
    basic_info_dict = dict(
        all_cov_names=[COVARIATE],
        col_t='days',
        col_group='location',
        predict_space=ln_gaussian_pdf,
        col_obs_compare='ln asddr',
        peaked_groups=peaked_groups
    )
    basic_model_dict = dict(
        param_names=['alpha', 'beta', 'p'],
        col_covs=[['intercept'], [COVARIATE], ['intercept']],
        link_fun=[np.exp, lambda x: x, np.exp],
        var_link_fun=[lambda x: x, lambda x: x, lambda x: x]
    )

    # basic fit parameter
    dummy_gprior = [0.0, np.inf]
    dummy_uprior = [-np.inf, np.inf]
    zero_uprior = [0.0, 0.0]
    fe_init = np.array([-2.5, 28.0, -8.05])
    initial_scalars = [[-0.5, 0.0, 0.5], [0.0], [0.0]]
    fe_bounds = [[-np.inf, 0.0], [15.0, 100.0], [-15, -6]]
    options = {
        'ftol': 1e-10,
        'gtol': 1e-10,
        'maxiter': 500,
        'disp': False
    }
    basic_fit_dict = dict(
        fe_init=fe_init,
        fe_bounds=fe_bounds,
        re_bounds=[zero_uprior]*3,
        fe_gprior=[dummy_gprior]*3,
        re_gprior=[dummy_gprior]*3,
        options=options
    )
    basic_joint_model_fit_dict = dict(
        fe_gprior=[dummy_gprior]*3,
        re_bounds=[dummy_uprior]*3,
        re_gprior=[dummy_gprior, [0.0, 10.0], dummy_gprior],
        smart_initialize=True,
        smart_init_options=options,
        options={
            'ftol': 1e-10,
            'gtol': 1e-10,
            'maxiter': 10,
            'disp': False
        }
    )

    # model control panel
    model_info_dict = {
        **deepcopy(basic_info_dict),
        'fun': ln_gaussian_cdf,
        'col_obs': 'ln ascdr',
        'col_obs_se':'obs_se',
        #'obs_se_func': lambda x: (1. / (1. + x)),
        'obs_se_func': None,
        'prior_modifier': lambda x: 10**(min(0.0, max(-1.0,
                    0.1*x - 1.5
        ))),
    }
    fit_dict = {
        **deepcopy(basic_fit_dict),
        'fun_gprior': [lambda params: params[0] * params[1], [np.exp(0.7), 1e-1]]
    }

    # prepare data (must exponentiate smoothed column, non-logged col is not smoothed)
    df['obs_se'] = 1 / (1 + df['days'])
    df.loc[df['pseudo'] == 1, 'obs_se'] = PSEUDO_SE
    df['Age-standardized death rate'] = np.exp(df['ln(age-standardized death rate)'])
    df = process_input(df, 'location_id', 'days', 'Age-standardized death rate',
                       col_covs=[COVARIATE, 'intercept', 'obs_se'])

    # set bounds on Gaussian mixture weights
    gm_bounds = np.repeat(np.array([[0, 2.]]), N_B, axis=0)
    gm_bounds = np.vstack([gm_bounds, [[0, 6.]]])  # add bounds on sum of weights
    gm_fit_dict = {
        'bounds': gm_bounds
    }

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## RUN MODEL
    # The Alpha Prior Model (flat asymmetric module)
    model = APFlatAsymmetricModel(
        beta_stride=2,
        mixture_size=N_B,
        daily_col='asddr',
        gm_fit_threshold=DATA_THRESHOLD,
        gm_fit_dict=gm_fit_dict,
        all_data=df,
        **model_info_dict,
        joint_model_fit_dict=basic_joint_model_fit_dict,
        basic_model_dict=basic_model_dict,
        fit_dict=fit_dict
    )
    model.run(
        n_draws=n_draws,
        prediction_times=np.arange(pred_days),
        cv_lower_threshold=1e-4, cv_upper_threshold=1.,
        smoothed_radius=[5, 5], num_smooths=2, exclude_groups=exclude_groups,
        exclude_below=0,
        last_info={
            model_location:[fix_day, fix_point]
        },
        initial_scalars=initial_scalars
    )
    daily_draws = model.process_draws(np.arange(pred_days),
                                      last_info={
                                          model_location:[fix_day, fix_point]
                                      })

    # turn draws into reporting space - ln(cumulative death rate)
    cumulative_draws = {}
    for group in daily_draws.keys():
        # sort
        loc_daily_draws = daily_draws[group][1][np.argsort(daily_draws[group][1][:,-1]),:]

        # get start pred
        if group == model_location and fix_point is not None:
            last_obs = fix_point
        else:
            last_obs = model.models[group].obs[-1]

        # add last turn to cumulative and last
        cumulative_draws.update({
            group: (daily_draws[group][0],
                    np.log(np.exp(last_obs) + np.exp(loc_daily_draws).cumsum(axis=1)))
        })

    return model, cumulative_draws


def plot_location(location, location_name, covariate_val, tm, lm, model_instance, draw, population, pdf=None, pred_days=PRED_DAYS):
    # get past curve point estimates
    tight_curve_t = np.arange(pred_days)
    tight_curve = tm.predict(tight_curve_t, group_name=location)
    if model_instance is None:
        loose_curve_t = np.arange(pred_days)
        loose_curve = lm.predict(loose_curve_t, group_name=location)
        loose_color = 'firebrick'
    else:
        # replace loose curve with mixture mean
        loose_curve_t = np.arange(pred_days)
        loose_curve = model_instance.predict(loose_curve_t, ln_gaussian_cdf, location)
        loose_color = 'darkgrey'

    # set up plot space
    fig, ax = plt.subplots(2, 2, figsize=(16.5, 8.5))

    # ln(asdr)
    ax[0, 0].scatter(tm.t, tm.obs, c='dodgerblue', edgecolors='navy')
    ax[0, 0].plot(draw[0], draw[1].mean(axis=0), color='dodgerblue')
    ax[0, 0].plot(tight_curve_t, tight_curve, color='forestgreen')
    ax[0, 0].plot(loose_curve_t, loose_curve, color=loose_color)
    ax[0, 0].fill_between(draw[0],
                          np.quantile(draw[1], 0.025, axis=0),
                          np.quantile(draw[1], 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[0, 0].set_xlim(tm.t.min() * 0.9, tm.t.max() + 28)
    ax[0, 0].set_ylabel('ln(asdr)')

    # asdr
    ax[0, 1].scatter(tm.t, np.exp(tm.obs), c='dodgerblue', edgecolors='navy')
    ax[0, 1].plot(draw[0], np.exp(draw[1]).mean(axis=0), color='dodgerblue')
    ax[0, 1].plot(tight_curve_t, np.exp(tight_curve), color='forestgreen')
    ax[0, 1].plot(loose_curve_t, np.exp(loose_curve), color=loose_color)
    ax[0, 1].fill_between(draw[0],
                          np.quantile(np.exp(draw[1]), 0.025, axis=0),
                          np.quantile(np.exp(draw[1]), 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[0, 1].set_ylabel('asdr')
    ax[0, 1].set_xlim(0, draw[0].max())
    ax[0, 1].set_ylim(0, np.quantile(np.exp(draw[1]), 0.975, axis=0).max()*1.1)

    # deaths
    ax[1, 0].scatter(tm.t, np.exp(tm.obs)*population, c='dodgerblue', edgecolors='navy')
    ax[1, 0].plot(draw[0], np.exp(draw[1]).mean(axis=0)*population, color='dodgerblue')
    ax[1, 0].plot(tight_curve_t, np.exp(tight_curve)*population, color='forestgreen')
    ax[1, 0].plot(loose_curve_t, np.exp(loose_curve)*population, color=loose_color)
    ax[1, 0].fill_between(draw[0],
                          np.quantile(np.exp(draw[1])*population, 0.025, axis=0),
                          np.quantile(np.exp(draw[1])*population, 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[1, 0].set_ylabel('deaths')
    ax[1, 0].set_xlim(0, draw[0].max())
    ax[1, 0].set_ylim(0, np.quantile(np.exp(draw[1])*population, 0.975, axis=0).max() * 1.1)
    ax[1, 0].set_xlabel('days')

    # daily deaths
    ax[1, 1].scatter(tm.t[1:], np.exp(tm.obs)[1:]*population - np.exp(tm.obs)[:-1]*population,
                     c='dodgerblue', edgecolors='navy')
    ax[1, 1].plot(draw[0][1:], (np.exp(draw[1])[:, 1:]*population - np.exp(draw[1])[:, :-1]*population).mean(axis=0),
                  color='dodgerblue')
    ax[1, 1].plot(tight_curve_t[1:], np.exp(tight_curve)[1:]*population - np.exp(tight_curve)[:-1]*population,
                  color='forestgreen')
    ax[1, 1].plot(loose_curve_t[1:], np.exp(loose_curve)[1:]*population - np.exp(loose_curve)[:-1]*population,
                  color=loose_color)
    ax[1, 1].fill_between(draw[0][1:],
                          np.quantile(np.exp(draw[1])[:, 1:]*population
                                      - np.exp(draw[1])[:, :-1]*population, 0.025, axis=0),
                          np.quantile(np.exp(draw[1])[:, 1:]*population
                                      - np.exp(draw[1])[:, :-1]*population, 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[1, 1].set_ylabel('daily deaths')
    ax[1, 1].set_xlim(0, draw[0].max())
    ax[1, 1].set_xlabel('days')

    plt.suptitle(f'{location_name} - SD cov: {np.round(covariate_val, 2)}', y=1.00025)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig)
    else:
        plt.show()


def run_death_models():
    # args = argparse.Namespace(
    #     cov_file='/ihme/covid-19/deaths/dev/2020_05_01_Europe_debug/model_data_descartes_21/60373_covariate.csv', 
    #     covariate_effect='gamma', 
    #     data_file='/ihme/covid-19/deaths/dev/2020_05_01_Europe_debug/model_data_descartes_21/60373.csv', 
    #     last_day_file='/ihme/covid-19/deaths/dev/2020_05_01_Europe_debug/last_day.csv', 
    #     model_location_id=60373, 
    #     n_draws=333, 
    #     output_dir='/ihme/covid-19/deaths/dev/2020_05_01_Europe_debug/model_data_descartes_21/60373', 
    #     peaked_file='/ihme/covid-19/deaths/mobility_inputs/2020_04_20/peak_locs_april20_.csv'
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_location_id', help='id of location to which we are standardizing.', type=int
    )
    parser.add_argument(
        '--data_file', help='Name of location-standardized data file.', type=str
    )
    parser.add_argument(
        '--cov_file', help='Name of covariate file.', type=str
    )
    parser.add_argument(
        '--last_day_file', help='Name of last day of deaths file.', type=str
    )
    parser.add_argument(
        '--peaked_file', help='Name of peaked locations file.', type=str
    )
    parser.add_argument(
        '--output_dir', help='Where we are storing results.', type=str
    )
    parser.add_argument(
        '--covariate_effect', help='Whether covariate is acting on beta or gamma.', type=str
    )
    parser.add_argument(
        '--n_draws', help='How many samples to take.', type=int
    )
    args = parser.parse_args()

    logger.info(args)
    # read data
    df = pd.read_csv(args.data_file)
    cov_df = pd.read_csv(args.cov_file)
    
    # only keep if more than one data point is present
    keep_idx = df.groupby('location_id')['location_id'].transform('count') > 1
    df = df[keep_idx].reset_index(drop=True)

    # try setting floor for covariate
    cov_df.loc[cov_df[COVARIATE] < 0.75, COVARIATE] = 0.75

    # attach covs to data file
    df = pd.merge(df, cov_df[['location_id', COVARIATE]], how='left')
    if df[COVARIATE].isnull().any():
        missing_locs = df.loc[df[COVARIATE].isnull(), 'Location'].unique().tolist()
        print(f'The following locations are missing covariates: {", ".join(missing_locs)}')
        df = df.loc[~df[COVARIATE].isnull()]
    df = df.sort_values(['location_id', 'Days']).reset_index(drop=True)  # 'Country/Region',

    # encode location_id for more explicit str indexing in model
    df['location_id'] = '_' + df['location_id'].astype(str)

    # add intercept
    df['intercept'] = 1.0

    # identify covariate value for our location
    location_cov = cov_df.loc[cov_df['location_id'] == args.model_location_id,
                              COVARIATE].item()

    # get list of peaked locations
    peaked_df = pd.read_csv(args.peaked_file)
    peaked_df['location_id'] = '_' + peaked_df['location_id'].astype(str)

    # get true ln(dr) on last day
    last_day_df = pd.read_csv(args.last_day_file)
    last_day_df = last_day_df.loc[last_day_df['location_id'] == args.model_location_id]
    if last_day_df.empty:
        fix_point = None
        fix_day = None
    else:
        fix_point = last_day_df['ln(death rate)'].item()
        fix_day = last_day_df['Days'].item()

    ## run models
    model_seed = get_hash(f'_{args.model_location_id}')
    np.random.seed(model_seed)
    # AP model for data poor
    if len(df.loc[df['location_id'] == f'_{args.model_location_id}']) < DATA_THRESHOLD:
        logger.info('Running data poor model')
        # or df.loc[df['location_id'] == f'_{args.model_location_id}', 'Deaths'].max() < 5:
        #
        # are we using a beta or gamma covariate
        if args.covariate_effect == 'beta':
            fix_gamma = True
        elif args.covariate_effect == 'gamma':
            fix_gamma = False

        # alpha prior model (no flat top)
        tight_model, loose_model, draws = ap_model(
            df=df[['location_id', 'intercept', 'Days', 'pseudo', 'ln(age-standardized death rate)', COVARIATE]],
            model_location=f'_{args.model_location_id}',
            location_cov=location_cov,
            n_draws=args.n_draws,
            peaked_groups=peaked_df.loc[peaked_df['location_id'].isin(df['location_id'].unique().tolist()), 'location_id'].to_list(),
            exclude_groups=peaked_df.loc[peaked_df['Location'].str.startswith('Wuhan'), 'location_id'].unique().tolist(),
            fix_gamma=fix_gamma,
            fix_point=fix_point,
            fix_day=fix_day
        )
        model = 'AP'

        # # get point estimate
        # d = pd.to_datetime(cov_df.loc[cov_df['location_id'] == args.model_location_id, 'threshold_date'].item())
        # t = np.arange(PRED_DAYS)
        # if f'_{args.model_location_id}' in list(draws.keys()):
        #     loose_curve = loose_model.predict(t, group_name=location)
        # else:

        # asdr = np.exp(tight_model.predict(t, ln_gaussian_cdf, f'_{args.model_location_id}'))
        # asddr = asdr[1:] - asdr[:-1]
        # point_df = pd.DataFrame({
        #     'location_id':args.model_location_id,
        #     'Date':[d + timedelta(days=int(t_i)) for t_i in t[1:]],
        #     'Age-standardized death rate':asddr
        # })
    else: # AP model for data rich
        logger.info('Running data rich model.')
        tight_model, draws = ap_flat_asym_model(
            df=df[['location_id', 'intercept', 'Days', 'pseudo', 'ln(age-standardized death rate)', COVARIATE]],
            model_location=f'_{args.model_location_id}',
            n_draws=args.n_draws,
            peaked_groups=peaked_df.loc[peaked_df['location_id'].isin(df['location_id'].unique().tolist()), 'location_id'].to_list(),
            exclude_groups=peaked_df.loc[peaked_df['Location'].str.startswith('Wuhan'), 'location_id'].unique().tolist(),
            fix_point=fix_point,
            fix_day=fix_day
        )
        loose_model = tight_model  # just to plug into plot
        model = 'AP flat asymmetrical'

        # get point estimate
        d = pd.to_datetime(cov_df.loc[cov_df['location_id'] == args.model_location_id, 'threshold_date'].item())
        t = np.arange(PRED_DAYS)
        asdr = np.exp(tight_model.predict(t, ln_gaussian_cdf, f'_{args.model_location_id}'))
        asddr = asdr[1:] - asdr[:-1]
        point_df = pd.DataFrame({
            'location_id':args.model_location_id,
            'Date':[d + timedelta(days=int(t_i)) for t_i in t[1:]],
            'Age-standardized death rate':asddr
        })

    # only save this location and overall draws
    subset_draws = dict()
    for model_label in [f'_{args.model_location_id}', 'overall']:
        if model_label in list(draws.keys()):
            subset_draws.update({
                model_label: draws[model_label]
            })

    # store outputs
    # data
    df[['location_id', 'intercept', 'Days', 'pseudo', 'ln(age-standardized death rate)', COVARIATE]].to_csv(f'{args.output_dir}/data.csv', index=False)
    # loose
    if model == 'AP':
        logger.info('Writing loose models.')
        with open(f'{args.output_dir}/loose_models.pkl', 'wb') as fwrite:
            pickle.dump(loose_model.models, fwrite, -1)
        with open(f'{args.output_dir}/loose_model_fit_dict.pkl', 'wb') as fwrite:
            pickle.dump(loose_model.fit_dict, fwrite, -1)
    else:
        # GM data
        logger.info('Writing Gaussian mixture metadata')
        with open(f'{args.output_dir}/gaussian_mixtures.pkl', 'wb') as fwrite:
            pickle.dump(tight_model.gaussian_mixtures, fwrite, -1)
        point_df.to_csv(f'{args.output_dir}/gm_point_estimate.csv', index=False)
    # tight
    logger.info('Writing tight models')
    with open(f'{args.output_dir}/tight_models.pkl', 'wb') as fwrite:
        pickle.dump(tight_model.models, fwrite, -1)
    with open(f'{args.output_dir}/tight_model_fit_dict.pkl', 'wb') as fwrite:
        pickle.dump(tight_model.fit_dict, fwrite, -1)
    # subset draws
    logger.info('Writing draws')
    with open(f'{args.output_dir}/draws.pkl', 'wb') as fwrite:
        pickle.dump(subset_draws, fwrite, -1)

    # plot (special condition if using multiple Gaussian)
    if model == 'AP':
        model_instance = None
    else:
        model_instance = tight_model
    logger.info('Writing model fit plots.')
    with PdfPages(f'{args.output_dir}/model_fits.pdf') as pdf:
        for location in tight_model.models.keys():
            location_name = df.loc[df['location_id'] == location, 'Location'].values[0]
            plot_location(location=location,
                          location_name=location_name,
                          covariate_val=cov_df.loc[cov_df['Location'] == location_name, COVARIATE].item(),
                          tm=tight_model.models[location],
                          lm=loose_model.models[location],
                          model_instance=model_instance,
                          draw=draws[location],
                          population=df.loc[df['location_id'] == location, 'population'].values[0],
                          pdf=pdf)


if __name__ == '__main__':
    run_death_models()
