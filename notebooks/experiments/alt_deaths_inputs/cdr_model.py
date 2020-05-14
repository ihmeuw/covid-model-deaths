import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def cdr_model(df: pd.DataFrame, dep_var_threshold: int, 
              daily: bool, log: bool, 
              dep_var: str, indep_vars: List[str],
              pdf) -> pd.DataFrame:
    # add intercept
    orig_cols = df.columns.to_list()
    df['intercept'] = 1

    # log transform, setting floor of 0.1 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.1 / df['population'].values[0]
    adj_vars = []
    for orig_var in [dep_var] + indep_vars:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df[mod_var] = np.log(df[mod_var])
            df.loc[df[orig_var] < floor, mod_var] = np.log(floor)
        # standardize
        if orig_var in indep_vars:
            df[mod_var] = (df[mod_var] - df[mod_var].mean()) / df[mod_var].std()
        adj_vars.append(mod_var)
    adj_dep_var = adj_vars[0]
    adj_indep_vars = ['intercept'] + adj_vars[1:]

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[adj_indep_vars].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = mod_df[adj_dep_var] >= dep_var_threshold
    non_na = ~mod_df[adj_dep_var].isnull()
    mod_df = mod_df.loc[above_thresh & non_na, [adj_dep_var] + adj_indep_vars].reset_index(drop=True)

    # run model and predict
    mod = sm.OLS(mod_df[adj_dep_var], mod_df[adj_indep_vars]).fit()
    df['Predicted death rate'] = mod.predict(df[adj_indep_vars])
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()
    
    param_df = pd.DataFrame(mod.params).reset_index()
    param_df.columns = ['variable', 'coefficient']

    # plot
    if pdf is not None:
        plotter(df, 
                [dep_var] + indep_vars,
                [adj_dep_var] + adj_indep_vars[1:],
                mod.params, 
                pdf)
    
    return df[orig_cols + ['Predicted death rate']]


def plotter(df: pd.DataFrame, unadj_vars: List[str], adj_vars: List[str], params: pd.Series, pdf=None):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, 3, figsize=(24, 16))

    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.5}
    smoothed_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    pred_lines = {'color':'forestgreen', 'alpha':0.75, 'linewidth':3}

    for i, (smooth_variable, model_variable) in enumerate(zip(unadj_vars, adj_vars)):
        # get coefficients (think of a more elegant way of doing this)
        if model_variable in params.index.to_list():
            param_label = f" - coefficient * 1e6: {np.round(params[model_variable] * 1e6, 3)}"
        else:
            param_label = ''
        
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        ax[0, i].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
        ax[0, i].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
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
    fig.suptitle(df['location_name'].values[0], y=1.0025, fontsize=14)
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)
    