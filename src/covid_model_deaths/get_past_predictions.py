import os
import time

import dill as pickle

import numpy as np
import pandas as pd

from datetime import datetime, timedelta


class Hindsight:
    def __init__(self, ensemble_dirs, location_name, location_id, model_used, population, tag='location_id'):
        # get our tagging of location_ids
        if tag == 'location_id':
            if not isinstance(location_id, str) or not location_id.startswith('_'):
                self.location_tag = f'_{location_id}'
            else:
                self.location_tag = location_id
        else:
            self.location_tag = location_name
        self.ensemble_dirs = ensemble_dirs
        self.location_name = location_name
        self.location_id = location_id
        self.obs_df = obs_df
        self.date_draws = date_draws
        self.population = population
        self.final_date = final_date
        
    def _collect_prediction(self, ensemble_dir):
        # read model outputs
        with open(f'{ensemble_dir}/{self.location_name}/loose_models.pkl', 'rb') as fread:
            loose_models = pickle.load(fread)
        