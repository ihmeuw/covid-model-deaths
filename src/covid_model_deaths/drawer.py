from datetime import datetime, timedelta
import os

import dill as pickle
import numpy as np
import pandas as pd


class Drawer:
    """Aggregates draws and stuff."""

    def __init__(self, ensemble_dirs, n_draws_list, location_name, location_id, peak_duration,
                 obs_df, date_draws, population, final_date='2020-10-01', tag='location_id'):
        # get our tagging of location_ids
        if tag == 'location_id':
            if not isinstance(location_id, str) or not location_id.startswith('_'):
                self.location_tag = f'_{location_id}'
            else:
                self.location_tag = location_id
        else:
            self.location_tag = location_name
        self.ensemble_dirs = ensemble_dirs
        self.n_draws_list = n_draws_list
        self.location_name = location_name
        self.location_id = location_id
        self.peak_duration = peak_duration
        self.obs_df = obs_df
        self.date_draws = date_draws
        self.population = population
        self.final_date = final_date

    def _collect_draws(self, ensemble_dir, n_draws, peak_days=3):
        # read model outputs
        if not os.path.exists(f'{ensemble_dir}/{self.location_id}/draws.pkl'):
            raise ValueError
        if os.path.exists(f'{ensemble_dir}/{self.location_id}/loose_models.pkl'):
            with open(f'{ensemble_dir}/{self.location_id}/loose_models.pkl', 'rb') as fread:
                models = pickle.load(fread)
        else:
            with open(f'{ensemble_dir}/{self.location_id}/tight_models.pkl', 'rb') as fread:
                models = pickle.load(fread)
        with open(f'{ensemble_dir}/{self.location_id}/draws.pkl', 'rb') as fread:
            model_draws = pickle.load(fread)

        # get predictions for given location, or use average if not present OR < 5 data points OR < 5 deaths
        if self.location_tag in list(model_draws.keys()):
            model_used = 'location'
            days = model_draws[self.location_tag][0]
            draws = model_draws[self.location_tag][1]
            past_pred = models[self.location_tag].predict(np.arange(days[0]), group_name=self.location_tag)
        else:
            # use overall
            model_used = 'overall'
            days = model_draws['overall'][0]
            draws = model_draws['overall'][1]
            past_pred = np.array([])

        if n_draws != draws.shape[0]:
            raise ValueError('Specified nubmer of draws different from actual number of draws.')

        # expand out by peak duration of peak days
        if self.peak_duration > 1:
            # draws = self._expand_peak(draws)
            raise ValueError('Attempting to expand peak, no longer supporting feature.')

        return model_used, days, draws, past_pred

    def _expand_peak(self, draws):
        # daily death rate space
        delta_draws = np.exp(draws[:, 1:]) - np.exp(draws[:, :-1])

        # find max of mean
        peak_idx = np.argmax(delta_draws.mean(axis=0))

        # expand that and stick into array (cutting off end to match expected length)
        peak = np.repeat(delta_draws[:, [peak_idx]], self.peak_duration, axis=1)
        if peak_idx == 0:
            delta_draws = np.hstack([peak,
                                     delta_draws[:, 1:-(self.peak_duration-1)]])
        else:
            delta_draws = np.hstack([delta_draws[:, :peak_idx],
                                     peak,
                                     delta_draws[:, peak_idx+1:-(self.peak_duration-1)]])

        # stick back on first obs, get cumulative, and convert back into log
        draws = np.log(
            np.hstack(
                [np.exp(draws[:, [0]]), delta_draws]
            ).cumsum(axis=1)
        )

        return draws

    def _get_dated_df(self, days, draws, past_pred):
        # apply dates to draws
        df = pd.concat([
            pd.DataFrame({
                'date': date,
                'draw': f'draw_{draw_n}',
                'deaths': np.exp(draw) * self.population
            })
            for draw_n, (date, draw)
            in enumerate(zip(np.vstack([self.date_draws + np.timedelta64(i, 'D') for i in days]).T, draws))
        ]).reset_index(drop=True)

        df = pd.pivot_table(df, index='date', columns='draw', values='deaths').reset_index()
        df['location_id'] = self.location_id
        df = df.sort_values('date').reset_index(drop=True)

        # apply dates to past
        if past_pred.size > 0:
            past_df = pd.concat([
                    pd.DataFrame({
                        'date': date,
                        'draw': f'draw_{draw_n}',
                        'deaths': np.exp(past_pred) * self.population
                    }) for draw_n, date in enumerate(
                        np.vstack([self.date_draws + np.timedelta64(i, 'D') for i in range(days[0])]).T
                    )
                ]
            ).reset_index(drop=True)
            past_df = pd.pivot_table(past_df, index='date', columns='draw', values='deaths').reset_index()
            past_df['location_id'] = self.location_id
            past_df = past_df.sort_values('date').reset_index(drop=True)
        else:
            past_df = pd.DataFrame(columns=df.columns)

        # get draw info and keep up to July 15th
        n_draws = len([i for i in df.columns if i.startswith('draw_')])
        draw_cols = [f'draw_{i}' for i in range(n_draws)]
        df = df.loc[df['date'] <= datetime.strptime(self.final_date, '%Y-%m-%d')].reset_index(drop=True)
        df = df.fillna(0)

        # sort draws
        draw_order = np.argsort(df.iloc[-1][draw_cols].values)
        sorted_draws = [f'draw_{i}' for i in draw_order]
        df = df.rename(index=str, columns=dict(zip(sorted_draws, draw_cols)))

        return df, past_df, n_draws, draw_cols

    def _fill_in_observed(self, df, n_draws, draw_cols):
        # fill in gaps in observed data
        filled_df = pd.DataFrame({
            'location_id': self.location_id,
            'Date': [self.obs_df['Date'].min() + timedelta(days=i)
                     for i in range((self.obs_df['Date'].max() - self.obs_df['Date'].min()).days + 1)]
        })
        filled_df = filled_df.merge(self.obs_df[['location_id', 'Date', 'Deaths']], how='left')
        filled_df = filled_df.sort_values('Date')
        filled_df['Deaths'] = filled_df['Deaths'].fillna(method='pad')
        filled_df = filled_df.rename(index=str, columns={'Date': 'date'})

        # expand to draws
        filled_df = pd.concat(
            [
                filled_df[['location_id', 'date']].reset_index(drop=True),
                pd.DataFrame(
                    np.repeat(np.expand_dims(filled_df['Deaths'].values, 0), n_draws, axis=0).T,
                    columns=draw_cols
                ).reset_index(drop=True)
            ],
            axis=1
        )

        # drop predictions before last day and slot in
        obs_end = filled_df['date'].max()
        filled_df['observed'] = True
        df = df.loc[df['date'] > obs_end]
        df['observed'] = False
        df = filled_df[['location_id', 'date', 'observed'] + draw_cols].append(
            df[['location_id', 'date', 'observed'] + draw_cols]
        ).reset_index(drop=True)

        return df

    def get_dated_draws(self):
        ensemble_draws = []
        ensemble_past = []
        for ensemble_dir, n_draws in zip(self.ensemble_dirs, self.n_draws_list):
            try:
                model_used, days, draws, past_pred = self._collect_draws(ensemble_dir, n_draws)
            except:
                print("No draws in ", ensemble_dir, self.location_name, self.location_id, ',dropping location.')
                raise ValueError
            ensemble_draws.append(draws)
            ensemble_past.append(past_pred)
        if not ensemble_draws:
            raise ValueError
        draws = np.vstack(ensemble_draws)
        past_pred = np.mean(ensemble_past, axis=0)
        ensemble_draws = dict(zip(self.ensemble_dirs, ensemble_draws))
        draw_df, past_df, n_draws, draw_cols = self._get_dated_df(days, draws, past_pred)
        if len(self.obs_df) > 0:
            draw_df = self._fill_in_observed(draw_df, n_draws, draw_cols)
        else:
            draw_df['observed'] = False
            draw_df = draw_df[['location_id', 'date', 'observed'] + draw_cols].reset_index(drop=True)
        draw_df['location'] = self.location_name
        past_df['location'] = self.location_name
        draw_df = draw_df[['location_id', 'location', 'date', 'observed'] + draw_cols].reset_index(drop=True)
        past_df = past_df[['location_id', 'location', 'date'] + draw_cols].reset_index(drop=True)

        return draw_df, past_df, model_used, days, ensemble_draws
