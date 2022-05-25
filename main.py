#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:08:31 2019

@author: laura
"""

#%% 

import numpy as np
import os
import glob
import json

# import tensorflow as tf
# tf.enable_eager_execution()
# tf.reset_default_graph()

from models import NNSolver, CJ_HJBSolver, CJ_HJBSolver_q32
from train import Solver
from get_config import Config, hexConfig
  
#%%

def main(configs):

    hex_conf = hexConfig().hex_config(configs)
    print("Configuration hex_code: ", hex_conf)

    # REMOVE PREVIOUS LOGS
    for file in glob.glob("logs/log_{}/*".format(hex_conf)):
        os.remove(file)
        
    # If no log directory exists, then create one:
    if not os.path.exists("logs/log_{}".format(hex_conf)):
        os.makedirs("logs/log_{}".format(hex_conf))       
        
    # SAVE DATA TO FILE
    minibatch_train_size = configs['minibatch_train_size']
    validation_size = configs['validation_size']
    n_iter = configs['n_iter']

    np.savez("logs/log_{}/data_SGD.npz".format(hex_conf),
             minibatch_train_size=minibatch_train_size,
             validation_size=validation_size,
             n_iter=n_iter)
    
    # MODEL DICTIONARY                 
    d = {"NN": NNSolver, 
         "LSTMCell": LSTMCellSolver, 
         "LSTM": LSTMSolver, 
         "HJB": CJ_HJBSolver,
         "HJB_q32": CJ_HJBSolver_q32
         }

    # DEFINE SOLVER TO USE
    model_config = configs['model']

    print("========== BEGIN SOLVER ==========")

    model = d[model_config]()  # init model
    solver = Solver(model, configs)  # create the solver using model and configs as inputs

    if model_config in ("HJB", "HJB_q32"):
        solver.control()  # output the control nu_t using CJ
    else:
        solver.train(n_iter)  # train the NN

    print("========== END SOLVER MKV FBSDE ==========")

    configs['alpha'] = solver.alpha  # can I erase these two lines? test!
    configs['kappa'] = solver.kappa
    configs['phi'] = solver.phi
    configs['A'] = solver.A
    
    # SAVE OUTPUT PATH TO FILE
    np.savez("logs/log_{}/data_solution_final.npz".format(hex_conf),
             X_path=solver.X_path,
             X1_path=solver.X1_path,
             X2_path=solver.X2_path,
             S_path=solver.S_path,
             Q_path=solver.Q_path,
             Control_path=solver.control_path,
             **configs)


if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    tf.reset_default_graph()

    # get meta configuration file
    with open("config_files/meta_config.json", "r") as read_file:
        meta_config_dict = json.load(read_file)

    # update meta configuration file using cmd line and model-specific configurations
    configs = Config().get_config(meta_config_dict)

    # run the model
    main(configs)
