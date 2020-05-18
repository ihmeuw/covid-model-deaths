import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from slime.core import MRData
from slime.model import CovModel, CovModelSet, MRModel
from smoother import smoother

def slimer(mod_df: pd.DataFrame, 
           death_var: str, case_var: str, test_var: str,
           log: bool, pred_df: pd.DataFrame):
    # model data
    mod_df['group'] = 0
    if log:
        col_covs = ['intercept', case_var, test_var]
    else:
        col_covs = [case_var, test_var]
    mrdata = MRData(mod_df,
                    col_group='group',
                    col_obs=death_var,
                    col_covs=col_covs)
    
    # set up covariates
    case_cov_model = CovModel(case_var,
                              use_re=False,
                              bounds=np.array([0., np.inf]))
    test_cov_model = CovModel(test_var,
                              use_re=False,
                              bounds=np.array([-np.inf, 0.]))
    if log:
        int_cov_model = CovModel('intercept',
                                 use_re=False,
                                 gprior=np.array([0, np.inf]))
        cov_models = CovModelSet([int_cov_model, case_cov_model, test_cov_model])
        variables = ['intercept', case_var, test_var]
    else:
        cov_models = CovModelSet([case_cov_model, test_cov_model])
        variables = [case_var, test_var]
    
    # run model
    model = MRModel(mrdata, cov_models)
    model.fit_model()
    model_params = model.result[0]
    
    # predict
    pred_df['Predicted death rate'] = 0
    for variable, param in zip(variables, model_params):
        pred_df[f'{variable} coefficient'] = param
        pred_df['Predicted death rate'] += pred_df[variable] * pred_df[f'{variable} coefficient']
    
    return pred_df
    
    
def cdr_model(df: pd.DataFrame, deaths_threshold: int, 
              daily: bool, log: bool, 
              death_var: str, case_var: str, test_var: str) -> pd.DataFrame:
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
            df.loc[df[mod_var] < floor, mod_var] = floor
            df[mod_var] = np.log(df[mod_var])
        adj_vars.update({orig_var:mod_var})

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = mod_df[death_var] * df['population'] >= deaths_threshold
    non_na = ~mod_df[adj_vars[death_var]].isnull()
    mod_df = mod_df.loc[above_thresh & non_na, ['intercept'] + list(adj_vars.values())].reset_index(drop=True)
    if len(mod_df) < 3:
        raise ValueError(f"Fewer than 3 days {deaths_threshold}+ deaths in {df['location_name'][0]}")

    # run model and predict
    df = slimer(
        mod_df, 
        adj_vars[death_var], adj_vars[case_var], adj_vars[test_var], 
        log,
        df
    )
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()
    
    return df
    

def synthesize_time_series(df: pd.DataFrame, 
                           daily: bool, log: bool, 
                           death_var: str, case_var: str, test_var: str,
                           n_draws: int = 1000, pdf=None) -> pd.DataFrame:
    # spline on output
    draw_df = smoother(df.copy().reset_index(drop=True), ['Death rate', 'Predicted death rate'], n_draws, daily, log)
    draw_cols = [col for col in draw_df.columns if col.startswith('draw_')]
    
    # add summary stats to dataset for plotting
    df = df.sort_values('Date').set_index('Date')
    draw_df = draw_df.sort_values('Date').set_index('Date')
    df['Smoothed predicted death rate'] = np.mean(draw_df[draw_cols], axis=1)
    df['Smoothed predicted death rate lower'] = np.percentile(draw_df[draw_cols], 2.5, axis=1)
    df['Smoothed predicted death rate upper'] = np.percentile(draw_df[draw_cols], 97.5, axis=1)
    df['Smoothed predicted daily death rate'] = np.nan
    df['Smoothed predicted daily death rate'][1:] = np.mean(np.diff(draw_df[draw_cols], axis=0), 
                                                            axis=1)
    df['Smoothed predicted daily death rate lower'] = np.nan
    df['Smoothed predicted daily death rate lower'][1:] = np.percentile(np.diff(draw_df[draw_cols], axis=0), 
                                                                        2.5, axis=1)
    df['Smoothed predicted daily death rate upper'] = np.nan
    df['Smoothed predicted daily death rate upper'][1:] = np.percentile(np.diff(draw_df[draw_cols], axis=0), 
                                                                        97.5, axis=1)
    df = df.reset_index()
    draw_df = draw_df.reset_index()
    first_day = df['Date'] == df.groupby('location_id')['Date'].transform(min)
    df.loc[first_day, 'Smoothed predicted daily death rate'] = df['Smoothed predicted death rate']
    df.loc[first_day, 'Smoothed predicted daily death rate lower'] = df['Smoothed predicted death rate lower']
    df.loc[first_day, 'Smoothed predicted daily death rate upper'] = df['Smoothed predicted death rate upper']
    
    # format draw data for infectionator
    draw_df = draw_df.rename(index=str, columns={'Date':'date'})
    draw_df[draw_cols] = draw_df[draw_cols] * draw_df[['population']].values
    del draw_df['population']
    
    # plot
    if pdf is not None:
        coef_cols = [i for i in df.columns if i.endswith('coefficient') if not i.startswith('intercept')]
        model_params = df[coef_cols].loc[0].to_dict()
        plotter(df, 
                [death_var, case_var, test_var],
                coef_cols,
                model_params, 
                pdf)
    
    return draw_df


def plotter(df: pd.DataFrame, unadj_vars: List[str], adj_vars: List[str], 
            model_params: dict, pdf):
    # set up plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))

    # aesthetic features
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.5}
    pred_lines = {'color':'forestgreen', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    for i, (smooth_variable, model_variable) in enumerate(zip(unadj_vars, [''] + adj_vars)):
        # get coefficients (think of a more elegant way of doing this)
        if model_variable in list(model_params.keys()):
            param_label = f" - coefficient: {np.round(model_params[model_variable], 8)}"
        else:
            param_label = ''
        
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        ax[0, i].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
        ax[0, i].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
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
        ax[1, i].axhline(0, color='black', alpha=0.25, linestyle='--')
        if 'death' in smooth_variable.lower():
            ax[1, i].set_xlabel('Date', fontsize=10)
        else:
            ax[1, i].set_xlabel('Date (+8 days)', fontsize=10)
        if i == 0:
            ax[1, i].set_ylabel('Daily', fontsize=10)

    # model prediction
    ax[0, 0].plot(df['Date'], df['Predicted death rate'] * df['population'], linestyle='--', **pred_lines)
    ax[1, 0].plot(df['Date'][1:], 
                  np.diff(df['Predicted death rate']) * df['population'][1:], 
                  **pred_lines)
    
    # smoothed
    ax[0, 0].plot(df['Date'], 
                  df['Smoothed predicted death rate'] * df['population'], linestyle='--', 
                  **smoothed_pred_lines)
    ax[0, 0].fill_between(
        df['Date'],
        df['Smoothed predicted death rate lower'] * df['population'], 
        df['Smoothed predicted death rate upper'] * df['population'], 
        **smoothed_pred_area
    )
    ax[1, 0].plot(df['Date'], 
                  df['Smoothed predicted daily death rate'] * df['population'], 
                  **smoothed_pred_lines)
    ax[1, 0].fill_between(
        df['Date'],
        df['Smoothed predicted daily death rate lower'] * df['population'], 
        df['Smoothed predicted daily death rate upper'] * df['population'], 
        **smoothed_pred_area
    )
        
    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=14)
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)
    