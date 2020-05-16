import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
# from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from ihme_math_utils.standard_errors import estimate_standard_error


def smoother(df: pd.DataFrame, smooth_var_sets: List[List[str]], 
             daily: bool, log: bool, n_draws: int = 1000) -> pd.DataFrame:
    # get overall knot options
    days = df.index.values
    day_knots = np.arange(days[0], days[-1], 7)[1:]
    
    for i, smooth_var_set in enumerate(smooth_var_sets):
        # extract inputs
        keep_idx = ~df[smooth_var_set].isnull().all(axis=1)
        y = df.loc[keep_idx, smooth_var_set].values
        if daily:
            y[1:] = np.diff(y, axis=0)
        if log:
            floor = 0.5 / df['population'][0]
            y[y < floor] = floor
            y = np.log(y)
        x = df.index[keep_idx].values
        
        if y[~np.isnan(y)].ptp() > 1e-10 and x.ptp() > 7:
            # create design matrix
            x_knots = tuple([i for i in day_knots if i in x])
            if x_knots[0] < x[0] + 3:
                x_knots = x_knots[1:]
            if x_knots[-1] > x[-1] - 3:
                x_knots = x_knots[:-1]
            x_dmat = dmatrix('cr(data, knots=data_knots)', {'data': x, 'data_knots':x_knots}, return_type='dataframe').values

            # get smoothed curve (dropping NAs)
            y_fit = y.flatten()
            x_dmat_fit = np.repeat(x_dmat, y.shape[1], axis=0)
            non_na_idx = ~np.isnan(y_fit)
            y_fit = y_fit[non_na_idx]
            x_dmat_fit = x_dmat_fit[non_na_idx]
            spline_mod = sm.GLM(y_fit, x_dmat_fit).fit()
            smooth_y = spline_mod.predict(x_dmat)
        else:
            # don't smooth if no difference
            smooth_y = y
            
        stds = estimate_standard_error(x=x.tolist(), y=y[:,1].tolist(), mode="fast", window_size=3)
        draws = np.random.normal(smooth_y, stds, (n_draws, smooth_y.size)).T
        #draws = np.sort(draws, axis=1)
        
        # back into linear cumulative and add prediction to data
        if log:
            draws = np.exp(draws)
        if daily:
            draws = draws.cumsum(axis=0)
        draw_df = df.loc[x, ['location_id', 'Date']].reset_index(drop=True)
        draw_df = pd.concat([draw_df, pd.DataFrame(draws, columns=[f'draw_{d}' for d in range(n_draws)])], axis=1)
            
    return draw_df
