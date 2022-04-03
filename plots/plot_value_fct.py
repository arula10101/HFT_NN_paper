#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:51:26 2020

@author: laura
"""

# %%

import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import cycle
import seaborn as sns
import numpy as np

from load_data import load_data
from value_function import using_lossfct
from solveODE import solveODEsystem
from value_function import control_dyn

import sys
sys.path.append("..")
tf.enable_eager_execution()

# %%

def mark_to_market(Q, S, X):
    mtm = (Q[:, 0] * S[:, 0]).reshape((len(Q[:, 0] * S[:, 0]), 1))
    sign = np.sign(Q[:, 0]).reshape((len(Q[:, 0]), 1))
    mtm_path = sign * (mtm - X)  # mtm = mark-to-market
    final_mtm = mtm_path[:, -1]  # + Q_path[:, -1] * S_path[:, -1]
    final_mtm_posQ0 = final_mtm[Q[:, 0] > 0]
    final_mtm_negQ0 = final_mtm[Q[:, 0] < 0]
    return final_mtm_posQ0, final_mtm_negQ0


# %% All cases are compared against the same pair (Q0,S0) coming from the standard NN

# NN standard case
NN_hex = '832c4088687ac6116d2ba6bcd556b609'
Q_path, S_path, X_path, X1_path, X2_path, ctrl_path, \
     model, stock_symbol, simulate, seasonality, \
     alpha, kappa, phi, A = load_data(NN_hex)
gamma = 2
NN_value_fct_dyn = using_lossfct(Q_path, S_path, X_path, phi, A, gamma)
NN_value_fct_dyn_mean = np.nanmean(NN_value_fct_dyn)
print("NN:", NN_value_fct_dyn_mean)
NN_final_mtm_posQ0, NN_final_mtm_negQ0 = mark_to_market(Q_path, S_path, X_path)

# HJB standard case using (Q0, S0) from NN
obj = solveODEsystem(kappa[0,0], alpha[0,0], phi, A, N_T=len(Q_path.T))
nu, Q, X, S = obj.control_dyn(Q_path, S_path, gamma)
HJB_value_fct_dyn, HJB_value_fct_dyn_mean = obj.valfct_dyn(Q, S, X)
print("HJB:", HJB_value_fct_dyn_mean)
HJB_final_mtm_posQ0, HJB_final_mtm_negQ0 = mark_to_market(Q, S, X)

# NN seasonality case
NN_seas_hex = 'be553552dcc561795873ff0b434611a7'
_, _, _, _, _, _, _, _, _, _, \
     alpha, kappa, _, _ = load_data(NN_seas_hex)
nu, Q, X, S = control_dyn(NN_seas_hex, Q_path, S_path, alpha, kappa, sigma=0.2)
NN_seas_value_fct_dyn = using_lossfct(Q, S, X, phi, A, gamma)
NN_seas_value_fct_dyn_mean = np.nanmean(NN_seas_value_fct_dyn)
print("NN seas:", NN_seas_value_fct_dyn_mean)
NN_seas_final_mtm_posQ0, NN_seas_final_mtm_negQ0 = mark_to_market(Q, S, X)

# NN seasonality and multipreference case
NN_multi_hex = '977640e7253c522207fec54ab7890baa'
_, _, _, _, _, _, _, _, _, _, \
     alpha, kappa, _, _ = load_data(NN_seas_hex)
nu, Q, X, S = control_dyn(NN_multi_hex, Q_path, S_path, alpha, kappa, A, phi, sigma=0.2)
NN_multi_value_fct_dyn = using_lossfct(Q, S, X, phi, A, gamma)
NN_multi_value_fct_dyn_mean = np.nanmean(NN_multi_value_fct_dyn)
print("NN multi:", NN_multi_value_fct_dyn_mean)
NN_multi_final_mtm_posQ0, NN_multi_final_mtm_negQ0 = mark_to_market(Q, S, X)

# HJB w/ gamma = 3/2 case using NN (Q0, S0), no seas
gamma=3/2
obj = solveODEsystem(kappa[0,0], alpha[0,0], phi, A, N_T=len(Q_path.T))
nu, Q, X, S = obj.control_dyn(Q_path, S_path, gamma)
HJB_32_value_fct_dyn, HJB_32_value_fct_dyn_mean = obj.valfct_dyn(Q, S, X)
print("HJB, gamma=3/2:", HJB_32_value_fct_dyn_mean)
HJB_32_final_mtm_posQ0, HJB_32_final_mtm_negQ0 = mark_to_market(Q, S, X)


# %%

NN_gamma32_hex = '47d1c7f1f934785cd685f6d684b322c2'
_, _, _, _, _, _, _, _, _, _, \
     alpha, kappa, phi, A = load_data(NN_gamma32_hex)
NN_32_value_fct_dyn = using_lossfct(Q_path, S_path, X_path, phi, A, gamma)
NN_32_value_fct_dyn_mean = np.nanmean(NN_value_fct_dyn)
print("NN, gamma=3/2:", NN_32_value_fct_dyn_mean)
NN_32_final_mtm_posQ0, NN_32_final_mtm_negQ0 = mark_to_market(Q_path, S_path, X_path)


# %%

labels = ["Stylized closed-form",
              "NNet on simulations",
              "NNet on simulations with seasonality",
              "Multi-preference NNet with seasonality",
              "NNet on simulation, $Q^{3/2}$ loss function"]

# %% Value Function vs Q_0

# Define plot, gamma=2
fig, ax = plt.subplots()
ax.xaxis.set_tick_params(rotation=30)

# Define linestyles to cycle
lines = ['--', '-.', ':']
linecycler = cycle(lines)
linestyle = next(linecycler)

# Plot KDEs
ax.set_title("Total Reward vs $Q_0$, $\gamma=2$")
ax.set_ylabel('Total Reward Function')
ax.set_xlabel('$Q_0$')
ax.axvline(x=0, color='k', linestyle='--')  # vertical line at 0
q0 = Q_path[:,0]
ax.scatter(q0, HJB_value_fct_dyn, c='red', marker=',', s=1, label=labels[0])
ax.scatter(q0, NN_value_fct_dyn, c='blue', marker=',', s=1, label=labels[1])
ax.scatter(q0, NN_seas_value_fct_dyn, c='green', marker=',', s=1, label=labels[2])
ax.scatter(q0, NN_multi_value_fct_dyn, c='lightpink', marker=',', s=1, label=labels[3])
ax.legend()
fig.savefig("valfct_2.pdf", bbox_inches='tight')
plt.show()

# %%

# Define plot gamma=3/2
fig, ax = plt.subplots()
ax.xaxis.set_tick_params(rotation=30)

# Define linestyles to cycle
lines = ['--', '-.', ':']
linecycler = cycle(lines)
linestyle = next(linecycler)

# Plot KDEs
ax.set_title("Total Reward vs $Q_0$, $\gamma=3/2$")
ax.set_ylabel('Total Reward')
ax.set_xlabel('$Q_0$')
ax.axvline(x=0, color='k', linestyle='--')  # vertical line at 0
q0 = Q_path[:,0]
ax.scatter(q0, HJB_32_value_fct_dyn, c='blue', marker=',', s=1, alpha=0.5, label=labels[0])
ax.scatter(q0, NN_32_value_fct_dyn, c='red', marker=',', s=1, alpha=0.5, label=labels[1])
ax.legend()
fig.savefig("valfct_32.pdf", bbox_inches='tight')
plt.show()


# %% MTM Wealth vs Q_0

# Define plot, gamma=2
fig, ax = plt.subplots()
ax.xaxis.set_tick_params(rotation=30)

# Define linestyles to cycle
lines = ['--', '-.', ':']
linecycler = cycle(lines)
linestyle = next(linecycler)

# Plot KDEs
ax.set_title("MTM Wealth vs $Q_0$, $\gamma=2$")
ax.set_ylabel('MTM Wealth')
ax.set_xlabel('$Q_0$')
ax.axvline(x=0, color='k', linestyle='--')  # vertical line at 0
q0_pos = Q_path[:,0][Q_path[:,0]>0]
ax.scatter(q0_pos, HJB_final_mtm_posQ0, c='blue', marker=',', s=1, alpha=0.5, label=labels[0])
ax.scatter(q0_pos, NN_final_mtm_posQ0, c='red', marker=',', s=1, alpha=0.5, label=labels[1])
ax.scatter(q0_pos, NN_seas_final_mtm_posQ0, c='green', marker=',', s=1, alpha=0.5, label=labels[2])
ax.scatter(q0_pos, NN_multi_final_mtm_posQ0, c='lightpink', marker=',', s=1, alpha=0.5, label=labels[3])
q0_neg = Q_path[:,0][Q_path[:,0]<0]
ax.scatter(q0_neg, HJB_final_mtm_negQ0, c='blue', marker=',', s=1, alpha=0.5)
ax.scatter(q0_neg, NN_final_mtm_negQ0, c='red', marker=',', s=1, alpha=0.5)
ax.scatter(q0_neg, NN_seas_final_mtm_negQ0, c='green', marker=',', s=1, alpha=0.5)
ax.scatter(q0_neg, NN_multi_final_mtm_negQ0, c='lightpink', marker=',', s=1, alpha=0.5)
ax.legend()
fig.savefig("mtm_2.pdf", bbox_inches='tight')
plt.show()

#%%

# Define plot, gamma=3/2
fig, ax = plt.subplots()
ax.xaxis.set_tick_params(rotation=30)

# Define linestyles to cycle
lines = ['--', '-.', ':']
linecycler = cycle(lines)
linestyle = next(linecycler)

# Plot KDEs
ax.set_title("MTM Wealth vs $Q_0$, $\gamma=3/2$")
ax.set_ylabel('MTM Wealth')
ax.set_xlabel('$Q_0$')
ax.axvline(x=0, color='k', linestyle='--')  # vertical line at 0
q0_pos = Q_path[:,0][Q_path[:,0]>0]
ax.scatter(q0_pos, HJB_32_final_mtm_posQ0, c='blue', marker=',', s=1, alpha=0.5, label=labels[0])
ax.scatter(q0_pos, NN_32_final_mtm_posQ0, c='red', marker=',', s=1, alpha=0.5, label=labels[1])
q0_neg = Q_path[:,0][Q_path[:,0]<0]
ax.scatter(q0_neg, HJB_32_final_mtm_negQ0, c='blue', marker=',', s=1, alpha=0.5)
ax.scatter(q0_neg, NN_32_final_mtm_negQ0, c='red', marker=',', s=1, alpha=0.5)
ax.legend()
fig.savefig("mtm_32.pdf", bbox_inches='tight')
plt.show()

