#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:52:58 2020

@author: laura
"""

#%%
import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from models import NNSolver
tf.enable_eager_execution()
np.random.seed(30)

#%% Calculate value function using two different methods:
# 1) Using h's and the expectation at time 0;
# 2) Using the problem dynamics and taking the expectation at the end.

def using_hs(Q, S, X, h0, h1, h2):
    value_fct_hs = X[:,0]  \
                    + S[:,0] * Q[:,0] \
                    + h2[0]*np.square(Q[:,0])/2 \
                    + h1[0]*Q[:,0] \
                    + h0[0]

#    value_fct_hs_mean = np.mean(value_fct_hs)
    return value_fct_hs



def using_lossfct(Q, S, X, phi, A, gamma):

    dt = 1/len(Q.T)
    value_fct_dyn = - phi * np.sum(np.power(np.abs(Q[:, :-1]), gamma)*dt, axis=1)  \
                    + X[:, -1] + Q[:, -1] * S[:,-1] - \
                        A * np.power(np.abs(Q[:, -1]), gamma)

#    value_fct_dyn_mean = np.nanmean(value_fct_dyn)
    return value_fct_dyn

# %%

def _create_stack(variable, n_samples):
    variable_stack = tf.reshape(tf.tile([tf.cast(variable, tf.float32)],
                                tf.stack([n_samples])),
                                [n_samples, 1])
    return variable_stack


def eval_NN_ckpt(model, optimizer, checkpoint_directory,
                 q,
                 avg_daily_volume=336115.75,  # comes from 'MRU', which is the stock we always use
                 N_t=78,
                 A=None):

    # Get checkpoint
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     optimizer_step=tf.train.get_or_create_global_step())
    ckpt = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    ckpt.assert_existing_objects_matched()

    # Evaluate NN on new input to obtain control
    nu_NN = np.zeros((N_t, len(q)))
    T = 1.
    dt = T/N_t
    N_q = len(q)
    q = q.reshape((len(q),1)) / avg_daily_volume

    for i_t in range(0, N_t):
        t_stack = _create_stack(T - i_t*dt, N_q)
        if A:
            A_stack = _create_stack(A, N_q)
            inputs = tf.concat([t_stack, q, A_stack], axis=1)
        else:
            inputs = tf.concat([t_stack, q], axis=1)
        output =  np.asarray(checkpoint.model(inputs))
        output = output.reshape((inputs.shape[0],))
        nu_NN[i_t, :] = output * avg_daily_volume

    return nu_NN



def control_dyn(hex_code, Q_path, S_path, alpha, kappa, A=None, phi=None, sigma=0.2):  # get full matrices Q and S

    avg_daily_volume = 336115.75  # comes from 'MRU', which is the stock we always use
    n_samples = len(Q_path)
    N_T = len(Q_path.T)
    dt = 1/N_T
    Q = np.zeros([n_samples, N_T])
    Q[:, 0] = Q_path[:, 0]
    X = np.zeros([n_samples, N_T])
    S = np.zeros([n_samples, N_T])
    S[:, 0] = S_path[:, 0]

    # get NN ckpt
    checkpoint_directory = "../checkpoints/checkpoint_{}".format(hex_code)
    model = NNSolver()
    optimizer = tf.train.AdamOptimizer(5e-4)
    #checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     optimizer_step=tf.train.get_or_create_global_step())
    ckpt = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    ckpt.assert_existing_objects_matched()

    nu_t = np.zeros((n_samples, N_T))

    for i in range(N_T-1):
        # MOVE FORWARD
        # Restore NN checkpoint and evaluate at Q[:,i]
        t_stack = _create_stack(1. - i*dt, n_samples)
        q = np.reshape(Q[:, i], (Q[:, i].shape[0], 1))
        q = q / avg_daily_volume

        if A and phi:
            A_stack = _create_stack(A, n_samples)
            phi_stack = _create_stack(phi, n_samples)
            inputs = tf.concat([t_stack, q, phi_stack, A_stack], axis=1)

        # inputs = tf.concat([t_stack, Q, phi_stack, A_stack], axis=1)
        else:
            inputs = tf.concat([t_stack, q], axis=1)

        output = np.asarray(checkpoint.model(inputs))
        output = output.reshape((inputs.shape[0],))
        nu_t[:, i] = output * avg_daily_volume

        dW = np.random.normal(0.0, 1.0, n_samples) * np.sqrt(dt)
        S[:, i+1] = S[:, i] + (alpha[0][i] * nu_t[:, i] * dt) + sigma * dW
        Q[:, i+1] = Q[:, i] + nu_t[:, i] * dt
        X[:, i+1] = X[:, i] - nu_t[:, i] * (S[:, i] + kappa[0][i] * nu_t[:, i]) * dt


    t_stack = _create_stack(1. - (i+1)*dt, n_samples)
    q = np.reshape(Q[:, -1], (Q[:, -1].shape[0], 1))
    q = q / avg_daily_volume

    if A:
        A_stack = _create_stack(A, n_samples)
        inputs = tf.concat([t_stack, q, phi_stack, A_stack], axis=1)
    else:
        inputs = tf.concat([t_stack, q], axis=1)

    output = np.asarray(checkpoint.model(inputs))
    output = output.reshape((inputs.shape[0],))
    avg_daily_volume = 336115.75  # comes from 'MRU', which is the stock we always use
    nu_t[:, -1] = output * avg_daily_volume

    return nu_t, Q, X, S






