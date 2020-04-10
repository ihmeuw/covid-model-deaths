import os
import sys
import argparse

from copy import deepcopy
import dill as pickle

import math
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import curvefit
from curvefit.test_pipline import APModel
from curvefit.utils import get_derivative_of_column_in_log_space, get_obs_se
from curvefit.utils import truncate_draws, data_translator, convex_combination

import warnings
warnings.filterwarnings('ignore')

RATE_THRESHOLD = -15  # should pass this in as argument
COVARIATE = 'cov_1w'


def death_model(df, model_location, location_cov, n_draws, peaked_groups, exclude_groups, pred_days=150):
    # our dataset
    df = df.copy()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SET UP
    # basic information and model setting
    basic_info_dict = dict(
        all_cov_names=[COVARIATE],
        col_t='Days',
        col_group='location_id',
        predict_space=curvefit.log_derf,
        col_obs_compare='d ln(age-standardized death rate)',
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
    fe_init = np.array([-3, 28.0, -8.05])
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

    # draw related paramters
    draw_dict = dict(
        n_draws=n_draws,
        prediction_times = np.arange(pred_days),
        cv_threshold=1e-4,
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
    obs_bounds = [25, np.inf] # filter the data rich models
    predict_cov = np.array([1.0, location_cov, 1.0]) # new covariates for the places.

    # tight prior control panel
    tight_info_dict = {
        **deepcopy(basic_info_dict),
        'fun': curvefit.log_erf,
        'col_obs': 'ln(age-standardized death rate)',
        'obs_se_func': lambda x: (1 / (1.0 + x)),
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
        'fun': curvefit.log_erf,
        'col_obs': 'ln(age-standardized death rate)',
        'obs_se_func': lambda x: (1 / (0.1 + x**1.4)),
        'prior_modifier': lambda x: 0.2
    }
    loose_fit_dict = {
        **deepcopy(basic_fit_dict),
        'fun_gprior': [lambda params: params[0] * params[1], dummy_gprior]
    }

    # prepare data
    df = get_derivative_of_column_in_log_space(
        df,
        col_t=basic_info_dict['col_t'],
        col_obs=tight_info_dict['col_obs'],
        col_grp=basic_info_dict['col_group']
    )
    df['daily deaths'] = np.exp(df['d ' + tight_info_dict['col_obs']])

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## RUN MODEL
    # The Alpha Prior Model
    tight_model = APModel(
        all_data=df,
        **tight_info_dict,
        joint_model_fit_dict=basic_joint_model_fit_dict,
        basic_model_dict=basic_model_dict,
        fit_dict=tight_fit_dict
    )
    #fe_gprior = tight_model.fit_dict['fe_gprior']
    #tight_model.fit_dict.update({
    #    'fe_gprior': [fe_gprior[0], [fe_gprior[1][0], 0.1], fe_gprior[2]]
    #})
    tight_model.run(**draw_dict)
    loose_model = APModel(
        all_data=df,
        **loose_info_dict,
        joint_model_fit_dict=basic_joint_model_fit_dict,
        basic_model_dict=basic_model_dict,
        fit_dict=loose_fit_dict
    )
    loose_model.run(**draw_dict)

    # get truncated draws
    tight_draws = tight_model.process_draws(draw_dict['prediction_times'])
    loose_draws = loose_model.process_draws(draw_dict['prediction_times'])
    combined_draws = {}
    for group in tight_draws.keys():
        draws = convex_combination(np.arange(tight_draws[group][1].shape[1]),
                                   tight_draws[group][1][np.argsort(tight_draws[group][1][:,-1]),:],
                                   loose_draws[group][1][np.argsort(loose_draws[group][1][:,-1]),:],
                                   basic_info_dict['predict_space'],
                                   start_day=start_day,
                                   end_day=end_day)
        combined_draws.update({
            group: (tight_draws[group][0],
                    np.log(np.exp(tight_model.models[group].obs[-1]) + np.exp(draws).cumsum(axis=1)))
        })

    # get overall draws
    filtered_tight_models = tight_model.run_filtered_models(
        df=tight_model.all_data, obs_bounds=obs_bounds
    )
    overall_tight_draws = tight_model.create_overall_draws(
        draw_dict['prediction_times'], filtered_tight_models, predict_cov, alpha_times_beta=alpha_times_beta,
        sample_size=draw_dict['n_draws'], slope_at=10, epsilon=draw_dict['cv_threshold']
    )
    filtered_loose_models = loose_model.run_filtered_models(
        df=loose_model.all_data, obs_bounds=obs_bounds
    )
    overall_loose_draws = loose_model.create_overall_draws(
        draw_dict['prediction_times'], filtered_loose_models, predict_cov, alpha_times_beta=alpha_times_beta,
        sample_size=draw_dict['n_draws'], slope_at=10, epsilon=draw_dict['cv_threshold']
    )

    # get specs and truncate overall, then combine
    if model_location in list(combined_draws.keys()):
        last_day = tight_model.models[model_location].t[-1]
        last_obs = tight_model.models[model_location].obs[-1]
        overall_time = draw_dict['prediction_times'][int(np.round(tight_model.models[model_location].t[-1])):]
    else:
        last_day = draw_dict['prediction_times'][0]
        last_obs = RATE_THRESHOLD
        overall_time = draw_dict['prediction_times']
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
                               overall_tight_draws[np.argsort(overall_tight_draws[:,-1]),:],
                               overall_loose_draws[np.argsort(overall_loose_draws[:,-1]),:],
                               basic_info_dict['predict_space'],
                               start_day=start_day,
                               end_day=end_day)
    combined_draws.update({
        'overall': (overall_time[1:],
                    np.log(np.exp(last_obs) + np.exp(draws).cumsum(axis=1)))
    })

    return tight_model, loose_model, combined_draws


def plot_location(location, location_name, covariate_val, tm, lm, draw, population, pdf=None, pred_days=150):
    # get past curve point estimates
    tight_curve_t = np.arange(pred_days)
    tight_curve = tm.predict(tight_curve_t, group_name=location)
    loose_curve_t = np.arange(pred_days)
    loose_curve = lm.predict(loose_curve_t, group_name=location)

    # set up plot space
    fig, ax = plt.subplots(2, 2, figsize=(16.5, 8.5))

    # ln(asdr)
    ax[0, 0].scatter(tm.t, tm.obs, c='dodgerblue', edgecolors='navy')
    ax[0, 0].plot(draw[0], draw[1].mean(axis=0), color='dodgerblue')
    ax[0, 0].plot(tight_curve_t, tight_curve, color='forestgreen')
    ax[0, 0].plot(loose_curve_t, loose_curve, color='firebrick')
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
    ax[0, 1].plot(loose_curve_t, np.exp(loose_curve), color='firebrick')
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
    ax[1, 0].plot(loose_curve_t, np.exp(loose_curve)*population, color='firebrick')
    ax[1, 0].fill_between(draw[0],
                          np.quantile(np.exp(draw[1])*population, 0.025, axis=0),
                          np.quantile(np.exp(draw[1])*population, 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[1, 0].set_ylabel('deaths')
    ax[1, 0].set_xlim(0, draw[0].max())
    ax[1, 0].set_ylim(0, np.quantile(np.exp(draw[1])*population, 0.975, axis=0).max()*1.1)
    ax[1, 0].set_xlabel('days')


    # daily deaths
    ax[1, 1].scatter(tm.t[1:], np.exp(tm.obs)[1:]*population - np.exp(tm.obs)[:-1]*population,
                     c='dodgerblue', edgecolors='navy')
    ax[1, 1].plot(draw[0][1:], (np.exp(draw[1])[:,1:]*population - np.exp(draw[1])[:,:-1]*population).mean(axis=0),
                  color='dodgerblue')
    ax[1, 1].plot(tight_curve_t[1:], np.exp(tight_curve)[1:]*population - np.exp(tight_curve)[:-1]*population,
                  color='forestgreen')
    ax[1, 1].plot(loose_curve_t[1:], np.exp(loose_curve)[1:]*population - np.exp(loose_curve)[:-1]*population,
                  color='firebrick')
    ax[1, 1].fill_between(draw[0][1:],
                          np.quantile(np.exp(draw[1])[:,1:]*population - np.exp(draw[1])[:,:-1]*population, 0.025, axis=0),
                          np.quantile(np.exp(draw[1])[:,1:]*population - np.exp(draw[1])[:,:-1]*population, 0.975, axis=0),
                          color='dodgerblue', alpha=0.25)
    ax[1, 1].set_ylabel('daily deaths')
    ax[1, 1].set_xlim(0, draw[0].max())
    #ax[1, 1].set_ylim(0, np.quantile(np.exp(draw[1])[:,1:]*population - np.exp(draw[1])[:,:-1]*population, 0.975, axis=0).max()*1.1)
    ax[1, 1].set_xlabel('days')

    plt.suptitle(f'{location_name} - SD cov: {np.round(covariate_val, 2)}', y=1.00025)
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(fig)
    else:
        plt.show()


def run_death_models():
    """
    args = argparse.Namespace(
        model_location='New York',
        model_location_id=555,
        data_file='/ihme/covid-19/deaths/prod/2020_04_07_US/model_data_equal_21/New York.csv',
        cov_file='/ihme/covid-19/deaths/prod/2020_04_07_US/model_data_equal_21/New York covariate.csv',
        peaked_file='/ihme/code/rmbarber/covid_19_ihme/final_peak_locs_04_07.csv',
        output_dir='/ihme/covid-19/deaths/prod/2020_04_07_US/model_data_equal_21/New York',
        n_draws=333
    )
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_location', help='Name of location to which we are standardizing.', type=str
    )
    parser.add_argument(
        '--model_location_id', help='id of location to which we are standardizing.', type=str
    )
    parser.add_argument(
        '--data_file', help='Name of location-standardized data file.', type=str
    )
    parser.add_argument(
        '--cov_file', help='Name of covariate file.', type=str
    )
    parser.add_argument(
        '--peaked_file', help='Name of peaked locations file.', type=str
    )
    parser.add_argument(
        '--output_dir', help='Where we are storing results.', type=str
    )
    parser.add_argument(
        '--n_draws', help='How many samples to take.', type=int
    )
    args = parser.parse_args()

    # read data
    df = pd.read_csv(args.data_file)
    cov_df = pd.read_csv(args.cov_file)

    # try setting floor for covariate
    cov_df.loc[cov_df[COVARIATE] < 0.1, COVARIATE] = 0.1

    # encode location_id so we don't end up w/ indexing issues
    df['location_id'] = '_' + df['location_id'].astype(str)

    # attach covs to data file -- should check if scenario run is necessary for location, so
    # as not to be wasteful...
    df = pd.merge(df, cov_df[['Location', COVARIATE]], how='left')
    if df[COVARIATE].isnull().any():
        missing_locs = df.loc[df[COVARIATE].isnull(), 'Location'].unique().tolist()
        #raise ValueError(f'The following locations are missing covariates: {", ".join(missing_locs)}')
        print(f'The following locations are missing covariates: {", ".join(missing_locs)}')
        df = df.loc[~df[COVARIATE].isnull()]
    df = df.sort_values(['location_id', 'Days']).reset_index(drop=True)  # 'Country/Region',

    # add intercept
    df['intercept'] = 1.0

    # identify covariate value for our location
    location_cov = cov_df.loc[cov_df['Location'] == args.model_location,
                              COVARIATE].item()

    # don't let it be below 10 / 28
    if location_cov < 0.36:
        location_cov = 0.36

    # get list of peaked locations
    peaked_df = pd.read_csv(args.peaked_file)
    peaked_df['location_id'] = '_' + peaked_df['location_id'].astype(str)

    # run models
    np.random.seed(15243)
    tight_model, loose_model, draws = death_model(
        df[['location_id', 'Location', 'intercept', 'Days', 'ln(age-standardized death rate)', COVARIATE]],
        f'_{args.model_location_id}', location_cov, args.n_draws,
        peaked_groups=peaked_df.loc[peaked_df['location_id'].isin(df['location_id'].unique().tolist()), 'location_id'].to_list(),
        exclude_groups=peaked_df.loc[peaked_df['Location'] == 'Wuhan City, Hubei', 'location_id'].unique().tolist()
    )

    # store outputs
    df.to_csv(f'{args.output_dir}/data.csv', index=False)
    with open(f'{args.output_dir}/loose_models.pkl', 'wb') as fwrite:
        pickle.dump(loose_model.models, fwrite, -1)
    with open(f'{args.output_dir}/tight_models.pkl', 'wb') as fwrite:
        pickle.dump(tight_model.models, fwrite, -1)
    with open(f'{args.output_dir}/draws.pkl', 'wb') as fwrite:
        pickle.dump(draws, fwrite, -1)

    # plot
    with PdfPages(f'{args.output_dir}/model_fits.pdf') as pdf:
        for location in tight_model.models.keys():
            location_name = df.loc[df['location_id'] == location, 'Location'].values[0]
            plot_location(location=location,
                          location_name=location_name,
                          covariate_val=cov_df.loc[cov_df['Location'] == location_name, COVARIATE].item(),
                          tm=tight_model.models[location],
                          lm=loose_model.models[location],
                          draw=draws[location],
                          population=df.loc[df['location_id'] == location, 'population'].values[0],
                          pdf=pdf)


if __name__ == '__main__':
    run_death_models()
