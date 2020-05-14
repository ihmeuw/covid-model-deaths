import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def smoother(df: pd.DataFrame, smooth_vars: List[str], daily: bool, log: bool, pdf=None) -> pd.DataFrame:
    # fill empty days
    df = df.sort_values('Date').set_index('Date')
    df = df.asfreq('D', method='pad').reset_index()
    
    # get overall knot options
    days = df.index.values
    day_knots = np.arange(days[0], days[-1], 7)[1:]
    
    # set up plotting if we are doing that
    if pdf is not None:
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(2, len(smooth_vars), figsize=(16.5, 8.5))
    
    for i, smooth_var in enumerate(smooth_vars):
        # extract inputs
        x = df.index[~df[smooth_var].isnull()].values
        y = df.loc[~df[smooth_var].isnull(), smooth_var].values
        if daily:
            y[1:] = y[1:] - y[:-1]
        if log:
            floor = 0.1 / df['population'][0]
            y[y < floor] = floor
            y = np.log(y)

        # create design matrix
        x_knots = tuple([i for i in day_knots if i in x])
        if x_knots[0] < x[0] + 3:
            x_knots = x_knots[1:]
        if x_knots[-1] > x[-1] - 3:
            x_knots = x_knots[:-1]
        x_dmat = dmatrix('cr(data, knots=data_knots)', {'data': x, 'data_knots':x_knots}, return_type='dataframe').values
        
        # get smoothed curve and store
        spline_mod = sm.GLM(y, x_dmat).fit()
        smooth_y = spline_mod.predict()
        raw_y = df.loc[~df[smooth_var].isnull(), smooth_var].values
        if log:
            smooth_y = np.exp(smooth_y)
        if daily:
            smooth_y = smooth_y.cumsum()
            # # if fitting in daily, scale up so that we still sum to observed total?
            # smooth_y *= raw_y[-1] / smooth_y[-1]
        df.loc[x, f'Smoothed {smooth_var.lower()}'] = smooth_y
        
        if pdf is not None:
            ax[0, i].plot(x, plot_y)
            ax[0, i].plot(x, smooth_y)
            ax[0, i].set_ylabel(f'{smooth_var} (cumulative)')

            ax[1, i].plot(x[1:], plot_y[1:] - plot_y[:-1])
            ax[1, i].plot(x[1:], smooth_y[1:] - smooth_y[:-1])
            ax[1, i].set_ylabel(f'{smooth_var} (daily)')
        
    if pdf is not None:
        fig.suptitle(df['location_name'][0], y=1.0025)
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)
    
    return df
