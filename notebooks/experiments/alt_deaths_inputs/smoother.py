import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from mr_spline import SplineFit


def smoother(df: pd.DataFrame, smooth_var_set: List[str], 
             daily: bool, log: bool, n_draws: int = 1000) -> pd.DataFrame:
    # get overall knot options
    days = df.index.values
    day_knots = np.arange(days[0], days[-1], 7)[1:]
    
    # extract inputs
    keep_idx = ~df[smooth_var_set].isnull().all(axis=1)
    y = df.loc[keep_idx, smooth_var_set].values
    if daily:
        y[1:] = np.diff(y, axis=0)
    if log:
        floor = 0.1 / df['population'][0]
        y[y < floor] = floor
        y = np.log(y)
    x = df.index[keep_idx].values

    if y[~np.isnan(y)].ptp() > 1e-10 and x.ptp() > 7:
        # create design matrix
        x_knots = [i for i in day_knots if i in x]
        if x_knots[0] < x[0] + 3:
            x_knots = x_knots[1:]
        if x_knots[-1] > x[-1] - 3:
            x_knots = x_knots[:-1]
        x_knots = np.array([x[0]] + x_knots + [x[-1]]) / x.max()

        # get smoothed curve (dropping NAs)
        y_fit = y.flatten()
        x_fit = np.repeat(x, y.shape[1], axis=0)
        non_na_idx = ~np.isnan(y_fit)
        y_fit = y_fit[non_na_idx]
        x_fit = x_fit[non_na_idx]
        mr_mod = SplineFit(x_fit, y_fit, 
                           {'spline_knots': x_knots,
                            'spline_degree': 3})
        mr_mod.fit_spline()
        smooth_y = mr_mod.predict(x)
    else:
        # don't smooth if no difference
        smooth_y = y

    # get uncertainty
    smooth_y = np.array([smooth_y]).T
    residuals = y - smooth_y
    if not log:
        residuals /= smooth_y
    residuals = residuals[~np.isnan(residuals)]
    residuals = residuals.flatten()
    draws = np.random.choice(residuals, n_draws, replace=True)
    draws = np.sort(draws)
    draws = np.array([draws])
    if not log:
        draws = smooth_y * draws
    draws = smooth_y + draws

    # back into linear cumulative and add prediction to data
    if log:
        draws = np.exp(draws)
        smooth_y = np.exp(smooth_y)
    draws *= (smooth_y / np.array([draws.mean(axis=1)]).T)
    if daily:
        draws = draws.cumsum(axis=0)
    draw_df = df.loc[x, ['location_id', 'Date', 'population']].reset_index(drop=True)
    draw_df = pd.concat([draw_df, pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(n_draws)])], axis=1)
        
    return draw_df
