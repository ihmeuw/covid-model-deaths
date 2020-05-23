# -*- coding: utf -*-
"""
    Simple spline fitting class using mrbrt.
"""
import numpy as np
import pandas as pd
from mrtool import MRData
from mrtool import LinearCovModel
from mrtool import MRBRT
from typing import List, Dict


class SplineFit:
    """Spline fit class
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 spline_var: str,
                 indep_vars: List[str], 
                 spline_options: Dict = dict(),
                 scale_se: bool = True,
                 scale_se_power: float = 0.2,
                 scale_se_floor_pctile: float = 0.05,
                 observed_var: str = None, 
                 pseudo_se_multiplier: float = 1.):
        # set up model data
        data = data.copy()
        if scale_se:
            data['obs_se'] = 1./np.exp(data[dep_var])**scale_se_power
            se_floor = np.percentile(data['obs_se'], scale_se_floor_pctile)
            data.loc[data['obs_se'] < se_floor, 'obs_se'] = se_floor
        else:
            data['obs_se'] = 1
        if observed_var:
            if not data[observed_var].dtype == 'bool':
                raise ValueError(f'Observed variable ({observed_var}) is not boolean.')
            data.loc[~data[observed_var], 'obs_se'] *= pseudo_se_multiplier

        # create mrbrt object
        data['study_id'] = 1
        mr_data = MRData(
            df=data,
            col_obs=dep_var,
            col_obs_se='obs_se',
            col_covs=indep_vars + [spline_var],
            col_study_id='study_id'
        )
        
        # cov models
        cov_models = []
        if 'intercept' in indep_vars:
            cov_models += [LinearCovModel(
                alt_cov='intercept',
                use_re=True,
                prior_gamma_uniform=np.array([0.0, 0.0]),
                name='intercept'
            )]
        if 'Model testing rate' in indep_vars:
            cov_models += [LinearCovModel(
                alt_cov='Model testing rate',
                use_re=False,
                prior_beta_uniform=np.array([-np.inf, 0.]),
                name='Model testing rate'
            )]
        if any([i not in ['intercept', 'Model testing rate'] for i in indep_vars]):
            bad_vars = [i for i in indep_vars if i not in ['intercept', 'Model testing rate']]
            raise ValueError(f"Unsupported independent variable(s) entered: {'; '.join(bad_vars)}")

        # spline cov model
        spline = LinearCovModel(
            alt_cov=spline_var,
            use_re=False,
            use_spline=True,
            **spline_options,
            name=spline_var
        )
        cov_models += [spline]
        
        # var names
        self.indep_vars = [i for i in indep_vars if i != 'intercept']
        self.spline_var = spline_var
        
        # model
        self.mr_model = MRBRT(mr_data, cov_models=cov_models)
        self.spline = spline.create_spline(mr_data)
        self.coef_dict = None

    def fit_model(self):
        self.mr_model.fit_model(inner_max_iter=30)
        self.coef_dict = {}
        for variable in self.indep_vars:
            self.coef_dict.update({
                variable: self.mr_model.beta_soln[self.mr_model.x_vars_idx[variable]]
            })
        spline_coefs = self.mr_model.beta_soln[self.mr_model.x_vars_idx[self.spline_var]]
        if 'intercept' in self.mr_model.linear_cov_model_names:
            intercept_coef = self.mr_model.beta_soln[self.mr_model.x_vars_idx['intercept']]
            spline_coefs = np.hstack([intercept_coef, intercept_coef + spline_coefs])
        self.coef_dict.update({
            self.spline_var:spline_coefs
        })
        
    def predict(self, pred_data: pd.DataFrame):
        preds = []
        for variable, coef in self.coef_dict.items():
            if variable == self.spline_var:
                mat = self.spline.design_mat(pred_data[variable].values,
                                             l_extra=True, r_extra=True)
            else:
                mat = pred_data[[variable]].values
            preds += [mat.dot(coef)]
        return np.sum(preds, axis=0)