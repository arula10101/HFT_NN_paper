#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:52:57 2020

@author: laura
"""


# %%

import os
import numpy as np


#%%
             
def load_data(hex_code):
    """ Load data output from neural network models, and HJB control.
    
    input: hex_code that encodes the dictionary of parameters
    output: path processes (Q, S, X, nu/control);
            simulation parameters (model, stock_symbol, simulate, seasonality);
            model parameters (alpha, kappa, phi, A)
    """
    # Get file name for last saved result from hex_code
    file = "../logs/log_{}/data_solution_final.npz".format(hex_code)
    file = os.path.join(os.path.dirname(__file__), file)
    # Get results
    with np.load(file) as results:
        Q_path = results['Q_path']
        S_path = results['S_path']
        X_path = results['X_path']
        X1_path = results['X1_path']
        X2_path = results['X2_path']
        ctrl_path = results['Control_path']  # NN control path
        model = results['model']

        simulate = results['simulate']
        seasonality = results['seasonality']
        alpha = results['alpha']
        kappa = results['kappa']
        phi = results['phi']
        A = results['A']

        try:
            stock_symbol = results['stock_symbol']
        except:
            stock_symbol = None

    return Q_path, S_path, X_path, X1_path, X2_path, ctrl_path, \
           model, stock_symbol, simulate, seasonality, \
           alpha, kappa, phi, A
           
       
# %%

def load_data_from_file(filename):
    """ Load data output from neural network models, and HJB control.

    input: file name for saved outputs
    output: path processes (Q, S, X, nu/control);
            simulation parameters (model, stock_symbol, simulate, seasonality);
            model parameters (alpha, kappa, phi, A)
    """
    # Get file name for last saved result from file name
    file = os.path.join(os.path.dirname(__file__), filename)

    # Get results
    with np.load(file) as results:
        Q_path = results['Q_path']
        S_path = results['S_path']
        X_path = results['X_path']
        ctrl_path = results['Control_path']  # NN control path

        alpha = results['alpha']
        kappa = results['kappa']

    return Q_path, S_path, X_path, ctrl_path, alpha, kappa
