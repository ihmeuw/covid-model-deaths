import os

from db_queries import get_location_metadata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_rows = 999
pd.options.display.max_columns = 99

#####
# EXAMPLE USE
# from plot_death_model import run_plots
# run_plots(DATESTAMP_LABEL, version = 1)
#####


def plot_death_model(draw_df: pd.DataFrame, location_id: int, loc_table: pd.DataFrame,
                     log: bool = False, show: bool = True):
    location = loc_table.loc[loc_table['location_id'] == location_id, 'Location'].item()
    country = loc_table.loc[loc_table['location_id'] == location_id, 'Country/Region'].item()
    f = plt.figure(figsize=(11, 8.5))
    plt.fill_between(
        draw_df.loc[draw_df['location_id'] == location_id, 'date'],
        np.percentile(draw_df.loc[draw_df['location_id'] == location_id,
                                  [i for i in draw_df.columns if i.startswith('draw_')]], 2.5, axis=1),
        np.percentile(draw_df.loc[draw_df['location_id'] == location_id,
                                  [i for i in draw_df.columns if i.startswith('draw_')]], 97.5, axis=1),
        alpha=0.25
    )
    plt.plot(
        draw_df.loc[draw_df['location_id'] == location_id, 'date'],
        draw_df.loc[draw_df['location_id'] == location_id,
                    [i for i in draw_df.columns if i.startswith('draw_')]].mean(axis=1),
    )
    plt.scatter(
        draw_df[draw_df['observed']].loc[draw_df['location_id'] == location_id, 'date'],
        draw_df[draw_df['observed']].loc[draw_df['location_id'] == location_id, 'draw_0']
    )
    plt.xticks(rotation=60)
    if log:
        plt.yscale('log')
        plt.ylabel("Cumulative Deaths - Log Scale")
    else:
        plt.ylabel("Cumulative Deaths")
    if location == country:
        plt.title(location)
    else:
        plt.title(location + ' - ' + country)
    if show:
        plt.show()
    return f


def make_euro_loc_table() -> pd.DataFrame:
    loc_df = get_location_metadata(location_set_id=35, gbd_round_id=6)
    loc_df = loc_df.loc[loc_df['level'] >= 3]

    # get regions, keep Europe
    loc_df['l2'] = loc_df['path_to_top_parent'].apply(lambda x: x.split(',')[2])
    euro_df = loc_df.loc[
        loc_df['l2'].isin(['42', '56', '73']) &
        ((loc_df['level'] == 3) | (loc_df['parent_id'] == 86)) &
        ~(loc_df['location_id'].isin([86, 81, 92])),
        ['location_id', 'location_name', 'level', 'parent_id']
    ]
    euro_df.loc[euro_df['level'] == 3, 'parent_id'] = euro_df['location_id']
    euro_df = euro_df.rename(index=str, columns={'location_name': 'Location'})
    loc_df = loc_df[['location_id', 'location_name']]
    loc_df = loc_df.rename(index=str, columns={'location_id': 'parent_id',
                                               'location_name': 'Country/Region'})
    euro_df = euro_df.merge(loc_df)

    ### NEED TO FIGURE OUT A BETTER WAY TO MANAGE LOCATIONS!!!
    # FIXME: WHAT IS THIS?
    esp_deu_df = pd.read_csv('/homes/aucarter/esp_deu_df.csv')

    euro_df = euro_df[['location_id', 'Location', 'Country/Region']].append(
        esp_deu_df[['location_id', 'Location', 'Country/Region']]
    ).reset_index(drop=True)

    euro_df['Location'] = euro_df['Location'].str.replace("'", "")
    return euro_df


def plot_aggregate(draw_df: pd.DataFrame):
    draw_cols = [i for i in draw_df.columns if i.startswith('draw_')]
    euro_deaths_df = draw_df.groupby('date', as_index=False)[draw_cols].sum()
    daily_deaths = euro_deaths_df[draw_cols].values[1:,:] - euro_deaths_df[draw_cols].values[:-1,:]
    euro_deaths_df = euro_deaths_df.iloc[1:]
    euro_deaths_df[draw_cols] = daily_deaths

    plt.figure(figsize=(11, 8.5))
    plt.fill_between(
        euro_deaths_df['date'],
        np.percentile(euro_deaths_df[draw_cols], 2.5, axis=1),
        np.percentile(euro_deaths_df[draw_cols], 97.5, axis=1),
        alpha=0.25
    )
    plt.plot(
        euro_deaths_df['date'],
        euro_deaths_df[draw_cols].mean(axis=1),
    )
    plt.title('Aggregate')
    plt.xticks(rotation=60)


# FIXME: Unused show variable, don't change apis till we know the calls.
def save_plots(in_path: str, out_dir: str, loc_table: pd.DataFrame, show: bool = True, agg: bool = True):
    # TODO: Don't do I/O here.
    draw_df = pd.read_csv(in_path)
    draw_df['date'] = draw_df['date'].map(pd.Timestamp)
    if agg:
        plot_aggregate(draw_df)
    # Iterate through locations in the input dataset
    for location_id in draw_df.location_id.unique():
        # Make figures
        f1 = plot_death_model(draw_df, location_id, loc_table)
        f2 = plot_death_model(draw_df, location_id, loc_table, log = True)

        # Make file name
        location = loc_table.loc[loc_table['location_id'] == location_id, 'Location'].item()
        country = loc_table.loc[loc_table['location_id'] == location_id, 'Country/Region'].item()
        if location == country:
            f_name = location
        else:
            f_name = country + '_' + location
        # Save
        f1_path = f'{out_dir}/{f_name}_deaths.pdf'
        f2_path = f'{out_dir}/{f_name}_deaths_log.pdf'
        f1.savefig(f1_path)
        f2.savefig(f2_path)


def run_plots(datestamp_label: str, version: str):
    date = datestamp_label[:10]

    in_path = f"/ihme/covid-19/deaths/dev/{datestamp_label}/euro_data.csv"
    out_dir = f'/snfs1/Project/covid/results/diagnostics/outputs_QC/deaths_{date}.{version}/europe/locs/'
    os.makedirs(out_dir, exist_ok=True)
    euro_locs = make_euro_loc_table()
    save_plots(in_path, out_dir, euro_locs)
