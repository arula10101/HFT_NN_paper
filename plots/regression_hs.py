#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:18:33 2020

@author: laura
"""

# %%

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# %% Regression to find h1 and h2


def regression_hs(Q_path, ctrl_path, alpha, kappa):
    """Regression of control onto the inventory.

    Q_path: numpy array, shape = (#days, #time_steps)
    alpha: numpy array, shape = (1, #time_steps)
    kappa: numpy array, shape = (1, #time_steps)

    """
    # Get NN h1 and h2:
    # Generate regression coefficient for each time step 't'
    h1 = []
    h2 = []
    r2 = []
    p_value_h1 = []
    p_value_h2 = []
    resid_var = []
    
    for t in range(len(Q_path.T)):  # number of 'time' steps
        q = Q_path.T[t]
        control = ctrl_path.T[t]
        data = pd.DataFrame({'q': q, 'control': control})
        reg = smf.ols('control ~ q', data=data).fit()
#        print('iteration' + str(t))
        # if t == 1:
        #     print(reg.summary())

        h1.append(reg.params.Intercept * 2*kappa[0][t])
        h2.append(reg.params.q*(2*kappa[0][t]) - alpha[0][t])
        r2.append(reg.rsquared)
        p_value_h1.append(reg.pvalues[0])
        p_value_h2.append(reg.pvalues[1])
        resid_var.append(np.var(reg.resid))

    return h1, h2, r2, p_value_h1, p_value_h2#, resid_var

# %%