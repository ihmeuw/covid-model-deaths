import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from smoother import smoother
from mr_spline import SplineFit

    
def cfr_model(df: pd.DataFrame, deaths_threshold: int, 
              daily: bool, log: bool, 
              dep_var: str, spline_var: str, indep_vars: List[str],
              model_dir: str) -> pd.DataFrame:
    # add intercept
    orig_cols = df.columns.to_list()
    df['intercept'] = 1

    # log transform, setting floor of 0.01 per population
    df = df.sort_values('Date').reset_index(drop=True)
    floor = 0.01 / df['population'].values[0]
    adj_vars = {}
    for orig_var in [dep_var, spline_var] + indep_vars:
        mod_var = f'Model {orig_var.lower()}'
        df[mod_var] = df[orig_var]
        if daily:
            start_idx = df.loc[~df[mod_var].isnull()].index.values[0]
            df[mod_var][start_idx+1:] = np.diff(df[mod_var].values[start_idx:])
        if log:
            df.loc[df[mod_var] < floor, mod_var] = floor
            df[mod_var] = np.log(df[mod_var])
        adj_vars.update({orig_var:mod_var})
    df['Model log'] = log
    df['Model daily'] = daily

    # keep what we can use to predict (subset further to fitting dataset below)
    non_na = ~df[list(adj_vars.values())[1:]].isnull().any(axis=1)
    df = df.loc[non_na].reset_index(drop=True)

    # lose NAs in deaths as well for modeling
    mod_df = df.copy()
    above_thresh = (mod_df[dep_var] * df['population']) >= deaths_threshold
    has_x = (mod_df[spline_var] * df['population']) >= 1
    non_na = ~mod_df[adj_vars[dep_var]].isnull()
    mod_df = mod_df.loc[above_thresh & has_x & non_na, ['intercept'] + list(adj_vars.values())].reset_index(drop=True)
    if len(mod_df) < 3:
        raise ValueError(f"Fewer than 3 days {deaths_threshold}+ deaths and 1+ cases in {df['location_name'][0]}")

    # run model and predict
    if (df[dep_var] * df['population']).max() > 20:
        x_knots = np.array([0., 0.33, 0.67, 1.])
        x_knots_idx = np.unique(np.quantile(mod_df[adj_vars[spline_var]], 
                                            x_knots),
                                return_index=True)[1]
        x_knots = x_knots[x_knots_idx]
    else:
        x_knots = np.array([0., 0.5, 1.])

    if x_knots.size > 3:
        # cubic spline with linear tails
        spline_options={
                'spline_knots': x_knots,
                'spline_knots_type': 'frequency',
                'spline_degree': 3,
                'spline_r_linear':True,
                'spline_l_linear':True,
                'prior_beta_uniform':np.array([0., np.inf]),
            }
    else:
        # linear spline
        spline_options={
                'spline_knots': x_knots,
                'spline_knots_type': 'frequency',
                'spline_degree': 1,
                'prior_beta_uniform':np.array([0., np.inf]),
            }
    if not daily:
        spline_options.update({'prior_spline_monotonicity':'increasing'})
    mr_mod = SplineFit(
        data=mod_df, 
        dep_var=adj_vars[dep_var],
        spline_var=adj_vars[spline_var],
        indep_vars=['intercept'] + list(map(adj_vars.get, indep_vars)),
        spline_options=spline_options,
        scale_se=False
    )
    mr_mod.fit_model()
    df['Predicted model death rate'] = mr_mod.predict(df)
    df['Predicted death rate'] = df['Predicted model death rate']
    if log:
        df['Predicted death rate'] = np.exp(df['Predicted death rate'])
    if daily:
        df['Predicted death rate'] = df['Predicted death rate'].cumsum()
    
    with open(f"{model_dir}/{df['location_id'][0]}.pkl", 'wb') as fwrite:
        pickle.dump(mr_mod, fwrite, -1)
    
    return df
    

def synthesize_time_series(df: pd.DataFrame, 
                           daily: bool, log: bool, 
                           dep_var: str, spline_var: str, indep_vars: List[str],
                           n_draws: int = 1000, plot_dir: str =None) -> pd.DataFrame:
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
    if plot_dir is not None:
        plotter(df, 
                [dep_var, spline_var] + indep_vars,
                f"{plot_dir}/{df['location_id'][0]}.pdf")
    
    return draw_df


def plotter(df: pd.DataFrame, unadj_vars: List[str], plot_file: str):
    # set up plot
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2, len(unadj_vars), figsize=(len(unadj_vars)*11, 16))

    # aesthetic features
    raw_lines = {'color':'navy', 'alpha':0.5, 'linewidth':3}
    raw_points = {'c':'dodgerblue', 'edgecolors':'navy', 's':100, 'alpha':0.5}
    pred_lines = {'color':'forestgreen', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_lines = {'color':'firebrick', 'alpha':0.75, 'linewidth':3}
    smoothed_pred_area = {'color':'firebrick', 'alpha':0.25}

    ax[0, 1].scatter(df['Confirmed case rate'], 
                     df['Death rate'], 
                     **raw_points)
    ax[0, 1].plot(df.loc[~df['Death rate'].isnull(), 'Confirmed case rate'], 
                  df.loc[~df['Death rate'].isnull(), 'Predicted death rate'], 
                  **pred_lines)
    ax[0, 1].set_xlabel('Cumulative case rate', fontsize=10)
    ax[0, 1].set_ylabel('Cumulative death rate', fontsize=10)
    
    for i, smooth_variable in enumerate(unadj_vars):
        # cumulative
        raw_variable = smooth_variable.replace('Smoothed ', '').capitalize()
        if 'death' in smooth_variable.lower():
            ax[0, i].plot(df['Date'], df[raw_variable] * df['population'], **raw_lines)
            ax[0, i].scatter(df['Date'], df[raw_variable] * df['population'], **raw_points)
            ax[1, i].set_xlabel('Date', fontsize=10)
            ax[0, i].set_ylabel(f'Cumulative {raw_variable.lower()}', fontsize=10)
            
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
        ax[1, i].set_ylabel(f'Daily {raw_variable.lower()}', fontsize=10)

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
    fig.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
    