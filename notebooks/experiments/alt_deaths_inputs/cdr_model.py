import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from slime.core import MRData
from slime.model import CovModel, CovModelSet, MRModel
from smoother import smoother

def slimer(mod_df: pd.DataFrame, 
           death_var: str, case_var: str, test_var: str,
           pred_df: pd.DataFrame):
    # model data
    mod_df['group'] = 0
    mrdata = MRData(mod_df,
                    col_group='group',
                    col_obs=death_var,
                    col_covs=['intercept', case_var, test_var])
    
    # set up covariates
    int_prior = mod_df[death_var].min()
    int_cov_model = CovModel('intercept',
                             gprior=np.array([0., np.inf]),
                             use_re=False)
    case_cov_model = CovModel(case_var,
                              use_re=False,
                              bounds=np.array([0., np.inf]))
    test_cov_model = CovModel(test_var,
                              use_re=False,
                              bounds=np.array([-np.inf, 0.]),)
    cov_models = CovModelSet([int_cov_model, case_cov_model, test_cov_model])
    
    # run model
    model = MRModel(mrdata, cov_models)
    model.fit_model()
    model_params = model.result[0]
    
    # predict
    pred_df['Predicted death rate'] = 0
    for variable, param in zip(['intercept', case_var, test_var], model_params):
        pred_df[f'{variable} coefficient'] = param
        pred_df['Predicted death rate'] += pred_df[variable] * pred_df[f'{variable} coefficient']
    
    return pred_df
    
    
def cdr_model(df: pd.DataFrame, death_threshold: int, 
              daily: bool, log: bool, smooth_results: bool,
              death_var: str, case_var: str, test_var: str,
              pdf) -> pd.DataFrame:
    # add intercept
    orig_cols = df.columns.to_list()
    df['intercept'] = 1

    # log transform, setting floor of 0.1 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.1 / df['population'].values[0]
    adj_vars = {}
    for orig_var in [death_var, case_var, test_var]:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df[mod_var] = np.log(df[mod_var])
            df.loc[df[orig_var] < floor, mod_var] = np.log(floor)
        adj_vars.update({orig_var:mod_var})

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = mod_df[adj_vars[death_var]] >= death_threshold
    non_na = ~mod_df[adj_vars[death_var]].isnull()
    mod_df = mod_df.loc[above_thresh & non_na, ['intercept'] + list(adj_vars.values())].reset_index(drop=True)

    # run model and predict
    df = slimer(
        mod_df, 
        adj_vars[death_var], adj_vars[case_var], adj_vars[test_var], 
        df
    )
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()
    model_params = df[[i for i in df.columns if i.endswith('coefficient')]].loc[0].to_dict()
    
    # smooth output?
    if smooth_results:
        df = smoother(df, ['Predicted death rate'], daily=False, log=False)

    # plot
    if pdf is not None:
        plotter(df, 
                [death_var, case_var, test_var],
                list(adj_vars.values()),
                model_params, 
                smooth_results,
                pdf)
    
    return df


def plotter(df: pd.DataFrame, unadj_vars: List[str], adj_vars: List[str], 
            model_params: dict, smooth_results: bool, pdf):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))

    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.5}
    smoothed_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    pred_lines = {'color':'forestgreen', 'alpha':0.75, 'linewidth':3}

    for i, (smooth_variable, model_variable) in enumerate(zip(unadj_vars, adj_vars)):
        # get coefficients (think of a more elegant way of doing this)
        if f'{model_variable} coefficient' in list(model_params.keys()):
            param_label = f" - coefficient: {np.round(model_params[f'{model_variable} coefficient'], 6)}"
        else:
            param_label = ''
        
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        ax[0, i].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
        ax[0, i].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
        if not smooth_results:
            ax[0, i].plot(df['Date'], df[smooth_variable] * df['population'], linestyle='--', **smoothed_lines)
        ax[0, i].set_title(f"{raw_variable.replace(' rate', '')}" + param_label, fontsize=12)
        if i == 0:
            ax[0, i].set_ylabel(f'Cumulative', fontsize=10)

        # daily
        ax[1, i].plot(df['Date'][1:], 
                      np.diff(df[raw_variable]) * df['population'][1:], 
                      **raw_lines)
        ax[1, i].scatter(df['Date'][1:], 
                         np.diff(df[raw_variable]) * df['population'][1:], 
                         **raw_points)
        if not smooth_results:
            ax[1, i].plot(df['Date'][1:], 
                          np.diff(df[smooth_variable]) * df['population'][1:], 
                          **smoothed_lines)
        ax[1, i].axhline(0, color='black', alpha=0.25, linestyle='--')
        ax[1, i].set_xlabel('Date', fontsize=10)
        if i == 0:
            ax[1, i].set_ylabel('Daily', fontsize=10)

    ax[0, 0].plot(df['Date'], df['Predicted death rate'] * df['population'], linestyle='--', **pred_lines)
    ax[1, 0].plot(df['Date'][1:], 
                  np.diff(df['Predicted death rate']) * df['population'][1:], 
                  **pred_lines)
    if smooth_results:
        ax[0, 0].plot(df['Date'], 
                      df['Smoothed predicted death rate'] * df['population'], linestyle='--', **smoothed_lines)
        ax[1, 0].plot(df['Date'][1:], 
                      np.diff(df['Smoothed predicted death rate']) * df['population'][1:], 
                      **smoothed_lines)
        
    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=14)
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)
    