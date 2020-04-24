import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib.backends.backend_pdf import PdfPages
from curvefit.core.model import CurveModel
import curvefit.core.utils as utils
from pprint import pprint


def prep_data(input_df, rate_col):
    print('Prepping data...')
    # Generate a date df for merging later
    date_df = input_df[['location_id','locaton_label', 'Date', 'Days']]
    date_df = date_df.rename(columns = {'Days':'days', 'location_id':'location'})
    model_df = utils.process_input(input_df, 'location_id', 'Days', rate_col)
    
    # Subset to locations with more than 3 observations
    obs_df = model_df.groupby('location').size()
    model_df = model_df.loc[model_df['ln asddr'] >= -35]
    model_df = model_df.loc[model_df['days'] > 0]
    model_df = model_df.loc[model_df['location'].isin(obs_df[obs_df > 3].index)]
    
    return model_df, date_df


def create_potential_peaks(model_df, input_df):
    print('Calculating potential peaks...')
    potential_peaked_groups, poly_fit = utils.create_potential_peaked_groups(
        model_df, 'location', 'days', 'ascdr',
        tol_num_obs=30,
        tol_after_peak=10,
        return_poly_fit=True
    )
    potential_peaked_names = sorted(input_df.loc[input_df['location_id'].isin(potential_peaked_groups), 'locaton_label'].unique())
    print(len(model_df['location'].unique()), '->', len(potential_peaked_groups))
    
    return potential_peaked_groups, potential_peaked_names, poly_fit


def compute_peak_day(peaked_locations, data, poly_fit, time_resolution=0.1):
    peak_day = {}
    for location in peaked_locations:
        df = data[location]
        c = poly_fit[location]
        peak_day.update({
            location: -0.5*c[1]/c[0]
        })
    return peak_day


def compute_peak_date(peaked_locations, data, poly_fit, time_resolution=0.1):
    peak_date = {}
    for location in peaked_locations:
        df = data[location]
        min_day = df['days'].min()
        min_date = df.loc[df.days == min_day].Date.item()
        c = poly_fit[location]
        peak = round(-0.5*c[1]/c[0])
        p_date = min_date + datetime.timedelta(days= (peak - min_day))
        peak_date.update({
            location: p_date.strftime('%Y-%m-%d')
        })
        
    return peak_date


def calc_peaks(date_df, model_df, potential_peaked_groups, poly_fit):
    df = model_df.merge(date_df)
    data = utils.split_by_group(df, 'location')
    peak_date = compute_peak_date(potential_peaked_groups, data, poly_fit, time_resolution=0.1)
    peak_day = compute_peak_day(potential_peaked_groups, data, poly_fit, time_resolution=0.1)
    
    return data, peak_date, peak_day


def plot_peak_and_mobility(loc_list, data, poly_fit, peak_date, peak_day, mob_df, measure_label, out_path=None):
    print('Plotting locations: ' + ', '.join(str(x) for x in loc_list))
    with PdfPages(out_path) as pdf:
        for location in loc_list:
            fig, ax = plt.subplots(1, 2, figsize=(16.5, 8.5))
            df_location = data[location]
            loc_name = df_location['locaton_label'][0]

            t = np.linspace(df_location['days'].min(), df_location['days'].max(), df_location['days'].max()+1) 
            log_y = np.polyval(poly_fit[location], t)
            y = np.exp(log_y)
            t = [df_location['Date'].min() - datetime.timedelta(days=int(t.min())) + datetime.timedelta(days=t_i) for t_i in t]

            t_peak = peak_day[location]
            log_y_peak = np.polyval(poly_fit[location], [t_peak])[0]
            y_peak = np.exp(log_y_peak)
            t_peak = pd.to_datetime(peak_date[location])

            ax[0].plot(mob_df.loc[mob_df['location_id'] == location, 'date'], 
                       mob_df.loc[mob_df['location_id'] == location, 'mobility'],
                       color='forestgreen', alpha=0.75)
            ax[0].plot([t_peak, t_peak], 
                       [mob_df.loc[mob_df['location_id'] == location, 'mobility'].min(), 
                        mob_df.loc[mob_df['location_id'] == location, 'mobility'].max()], 
                       'r--')
            ax[0].set_title(f'change in mobility')
            loc = plticker.MaxNLocator(7) # this locator puts ticks at regular intervals
            ax[0].xaxis.set_major_locator(loc)
            
            ax[1].scatter(df_location['Date'], df_location['asddr'], edgecolors='k', color='#ADD8E6')
            ax[1].plot(t, y, c='#4682B4')
            ax[1].scatter(t_peak, y_peak, edgecolors='k', color='r')
            #ax[1].set_ylim(-1, df_location['asddr'].max()*1.2)
            ax[1].plot([t_peak, t_peak], [0.0, y_peak], 'r--')
            ax[1].set_title(f'daily {measure_label}')
            loc = plticker.MaxNLocator(7) # this locator puts ticks at regular intervals
            ax[1].xaxis.set_major_locator(loc)
            
            plt.suptitle(loc_name, y=1.00025)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
    