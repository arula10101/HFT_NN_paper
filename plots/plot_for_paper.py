#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:38:08 2020

@author: laura
"""


# %%

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from itertools import cycle
import numpy as np

from load_data import load_data
from regression_hs import regression_hs

#%% 

def plot_new(list_of_hex_codes):
    
    # Define plot
    fig, ax = plt.subplots(2, 2, figsize=(10,8))

    # Define linestyles to cycle
    lines = ['solid','dotted','solid','solid','dotted']
    linecycler = cycle(lines)

    widths = [5, 4, 5, 2, 2]
    widthcycler = cycle(widths)

    colors = ['lightblue', 'grey', 'thistle', 'crimson', 'purple']
    colorcycler = cycle(colors)

    labels = ["Stylized closed-form",
              "NNet on simulations",
              "NNet on simulations with seasonality",
              "NNet on real data with seasonality",
              "Multi-preference NNet\non real data with seasonality"]
    
    #labels = ["$(A,\phi)=(0.5,0.1)$",
    #          "$(A,\phi)=(0.01,0.0007)$"]

    # Define isolated plots for paper
    fig0, ax0 = plt.subplots(1, 1, figsize=(6,4))
    fig1, ax1 = plt.subplots(1, 1, figsize=(6,4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,4))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6,4))

    prefix = "/Users/laura/Desktop/HFT_NN_paper/plots/"
    

    i=0
    for hex_code in list_of_hex_codes:
        label=labels[i]
        i+=1
        linestyle = next(linecycler)
        linewidth = next(widthcycler)
        # color_next = next(c)
        color_next = next(colorcycler)

        # Load data created with train.py
        Q_path, S_path, X_path, X1_path, X2_path, ctrl_path, \
             model, stock_symbol, simulate, seasonality, \
             alpha, kappa, phi, A = load_data(hex_code)

        ctrl_pos = np.mean(ctrl_path[Q_path[:, 0]>0], axis=0)
        ctrl_neg = np.mean(ctrl_path[Q_path[:, 0]<0], axis=0)
                   
        h1, h2, r2, pvalue_h1, pvalue_h2 = regression_hs(Q_path[:, :-1], ctrl_path[:, :-1], alpha, kappa)
        
        
        ax[0,0].set_title("$h_1$")
        ax[0,0].plot(h1, label=label, alpha=0.7, color=color_next, linestyle=linestyle)
        ax[0,0].set_ylim((-1, 1))
        
        ax0.set_title("$h_1$")
        ax0.plot(h1, label=label, alpha=0.7, color=color_next, linestyle=linestyle, linewidth=linewidth)
        ax0.set_ylim((-np.median(Q_path), np.median(Q_path)))
        ax0.set_xlabel('Time')
        ax0.legend(loc='lower right')
        fig0.savefig(prefix + "_h1.pdf", bbox_inches='tight')


        
        # Plot h2
        ax[0,1].set_title("$h_2$")
        ax[0,1].plot(h2, label=label, alpha=0.7, color=color_next, linestyle=linestyle)

        ax1.set_title("$h_2$")
        ax1.plot(h2, label=label, alpha=0.7, color=color_next, linestyle=linestyle, linewidth=linewidth)
        ax1.set_xlabel('Time')
        fig1.savefig(prefix + "_h2.pdf", bbox_inches='tight')


        # Plot R^2
        ax[1,0].set_title("$R^2$")
        ax[1,0].plot(r2, label=label, alpha=0.7, color=color_next, linestyle=linestyle)
        ax[1,0].set_xlabel('Time')
        ax[1,0].set_ylim((0, 1.05))
        
        ax2.set_title("$R^2$")
        ax2.plot(r2, label=label, alpha=0.7, color=color_next, linestyle=linestyle, linewidth=linewidth)
        ax2.set_xlabel('Time')
        # ax2.set_ylim((0, 1.05))
        fig2.savefig(prefix + "_r2.pdf", bbox_inches='tight')

        
        # Plot nu
        ax[1,1].set_title("Average Control Process (nu)")
        ax[1,1].plot(ctrl_pos/78, label=label, alpha=0.7, color=color_next, linestyle=linestyle)
        # ax[1,1].plot(ctrl_neg/78, alpha=0.7, color=color_next, linestyle=linestyle)
        ax[1,1].set_xlabel('Time')
        # ax[1,1].legend(bbox_to_anchor=(.4, 2.8))
        
        ax3.set_title("Average Control Process")
        # ax3.plot(ctrl_pos, label=label, alpha=0.7, color=color_next, linestyle=linestyle)
        ax3.plot(ctrl_neg, label=label, alpha=0.7, color=color_next, linestyle=linestyle, linewidth=linewidth)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Speed of trading\n(quantity per 5 minutes)')
        ax3.legend(loc='upper right')
        fig3.savefig(prefix + "ctrl.pdf", bbox_inches='tight')
        # ax3.legend(bbox_to_anchor=(1.1,.7))
        # ax3.legend(bbox_to_anchor=(1.5,.7))
        
if __name__ == "__main__":
    
    # How to use:    
    HJB_hex = '8431496b332739f5fea4bb07d92ded80'
    NN_hex = '832c4088687ac6116d2ba6bcd556b609'
    NN_seas = 'be553552dcc561795873ff0b434611a7'
    NN_real_seas_hex = '7b1c7c46748e623db9645d594e0dafff'
    multi_hex = '977640e7253c522207fec54ab7890baa'
    # # python3 main.py -m "NN" -sim "no" -seas "yes" -s "MRU" -N_T 78 -n 100001 -A 0.01 -phi 0.0007 -q_sig_t 10
    plot_new([HJB_hex,
               NN_hex,
               NN_seas,
               NN_real_seas_hex,
               multi_hex])

    #q32_linear = '1a9665528708def727a44745763d8d63'
    #q32_nonlinear = '47d1c7f1f934785cd685f6d684b322c2'
    #plot_new([q32_linear,
    #          q32_nonlinear])