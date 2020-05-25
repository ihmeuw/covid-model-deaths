import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from mr_spline import SplineFit


def smoother(df: pd.DataFrame, smooth_var_set: List[str], 
             n_draws: int, daily: bool, log: bool) -> pd.DataFrame:
    # get overall knot options (one week apart)
    days = df.index.values
    week_knots = np.arange(days[0], days[-1], 7)[1:]
    
    # extract inputs
    keep_idx = ~df[smooth_var_set].isnull().all(axis=1)
    no_na_idx = ~df[smooth_var_set].isnull().any(axis=1)
    y = df.loc[keep_idx, smooth_var_set].values
    if daily:
        y[1:] = np.diff(y, axis=0)
    if log:
        floor = 0.01 / df['population'][0]
        y[y < floor] = floor
        y = np.log(y)
    x = df.index[keep_idx].values

    if y[~np.isnan(y)].ptp() > 1e-10:
        # determine knots
        x_knots = np.percentile(x[no_na_idx], (5, 25, 50, 75, 95)).tolist()
        x_knots = np.array([x[0]] + x_knots + [x[-1]]) / x.max()

        # get smoothed curve (dropping NAs, inflating variance for deaths from cases - ASSUMES THAT IS SECOND COLUMN)
        obs_data = y.copy()
        obs_data[:,0] = 1
        obs_data[:,1] = 0
        y_fit = y.flatten()
        obs_data = obs_data.flatten()
        x_fit = np.repeat(x, y.shape[1], axis=0)
        non_na_idx = ~np.isnan(y_fit)
        y_fit = y_fit[non_na_idx]
        obs_data = obs_data[non_na_idx]
        x_fit = x_fit[non_na_idx]
        mod_df = pd.DataFrame({
            'y':y_fit,
            'intercept':1,
            'x':x_fit,
            'observed':obs_data
        })
        mod_df['observed'] = mod_df['observed'].astype(bool)
        spline_options={
                'spline_knots': x_knots,
                'spline_knots_type': 'domain',
                'spline_degree': 3,
                'spline_r_linear':True,
                'spline_l_linear':True
            }
        if not daily:
            spline_options.update({'prior_spline_monotonicity':'increasing'})
        mr_mod = SplineFit(
            data=mod_df, 
            dep_var='y',
            spline_var='x',
            indep_vars=['intercept'], 
            spline_options=spline_options,
            scale_se=daily,
            observed_var='observed',
            pseudo_se_multiplier=1.33
        )
        mr_mod.fit_model()
        smooth_y = mr_mod.predict(pd.DataFrame({'intercept':1, 'x': x}))
    else:
        # don't smooth if no difference
        smooth_y = y

    # get uncertainty
    smooth_y = np.array([smooth_y]).T
    residuals = y - smooth_y
    residuals = residuals[~np.isnan(residuals)]
    mad = np.median(np.abs(residuals))
    std = mad * 1.4826
    draws = np.random.normal(smooth_y, std, (smooth_y.size, n_draws))
    #draws = stats.t.rvs(dof, loc=smooth_y, scale=std, size=(smooth_y.size, n_draws))

    # set to linear, add up cumulative, and create dataframe
    if log:
        #draws -= draws.var(axis=1, keepdims=True) / 2
        draws = np.exp(draws)
        draws *= np.exp(smooth_y) / draws.mean(axis=1, keepdims=True)
    if daily:
        draws = draws.cumsum(axis=0)
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df['Smooth log'] = log
    draw_df['Smooth daily'] = daily
    draw_df = pd.concat([draw_df, pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(n_draws)])], axis=1)
        
    return draw_df
