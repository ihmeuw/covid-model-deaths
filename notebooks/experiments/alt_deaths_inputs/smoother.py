import pandas as pd
import numpy as np
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
        floor = 0.1 / df['population'][0]
        y[y < floor] = floor
        y = np.log(y)
    x = df.index[keep_idx].values

    if y[~np.isnan(y)].ptp() > 1e-10 and x.ptp() > 7:
        # determine knots
        x_knots = np.percentile(x[no_na_idx], (15, 50, 85)).tolist()
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
        mr_mod = SplineFit(x_fit, y_fit, obs_data,
                           {'spline_knots': x_knots,
                            'spline_degree': 3,
                            'spline_r_linear':True,
                            'spline_l_linear':True})
        mr_mod.fit_spline()
        smooth_y = mr_mod.predict(x)
    else:
        # don't smooth if no difference
        smooth_y = y

    # get uncertainty
    smooth_y = np.array([smooth_y]).T
    residuals = y - smooth_y
    residuals = residuals[~np.isnan(residuals)]
    mad = np.median(np.abs(residuals))
    std = mad * 1.4826
    draws = np.random.normal(0, std, (n_draws, smooth_y.size))
    draws = smooth_y + draws.T

    # set to linear, make sure mean of linear draws equals linear point estimate, add up cumulative, and create dataframe
    if log:
        draws = np.exp(draws)
        smooth_y = np.exp(smooth_y)
    draws *= (smooth_y / np.array([draws.mean(axis=1)]).T)
    if daily:
        draws = draws.cumsum(axis=0)
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df['Smooth log'] = log
    draw_df['Smooth daily'] = daily
    draw_df = pd.concat([draw_df, pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(n_draws)])], axis=1)
        
    return draw_df
