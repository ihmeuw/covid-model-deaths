import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style('whitegrid')


def plot_crude_rates(df: pd.DataFrame, level: str = None):
    """Plots crude (population level) rates."""
    # TODO: Generalize and switch to location ids.
    # TODO: Move to plotting module, auto-generate plots in a standard location
    #  during model runs.
    # subset if needed
    if level == 'subnat':
        df = df.loc[df['Location'] != df['Country/Region']].reset_index(drop=True)
    elif level == 'admin0':
        df = df.loc[df['Location'] == df['Country/Region']].reset_index(drop=True)
    elif level is not None:
        raise ValueError('Invalid level specified in plotting call.')

    # do the plotting
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    df = df.sort_values(['Country/Region', 'Location', 'Date'])
    for location, country in df[['Location', 'Country/Region']].drop_duplicates().values:
        # TODO: Make a rule or abstract to a country-style mapping.
        if country == 'China':
            style = '--'
        elif country == 'United States of America':
            style = ':'
        elif country == 'Italy':
            style = '-.'
        elif country == 'Spain':
            style = (0, (1, 10))
        elif country == 'Germany':
            style = (0, (5, 10))
        else:
            style = '-'
        plt.plot(df.loc[df['Location'] == location, 'Days'],
                 np.log(df.loc[df['Location'] == location, 'Death rate']),
                 label=location, linestyle=style)
    plt.xlabel('Days')
    plt.xlim(0, df['Days'].max())
    plt.ylabel('ln(death rate)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    plt.show()
