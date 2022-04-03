#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:18:53 2019

@author: laura
"""

import tensorflow as tf
import numpy as np
import pickle
from operator import itemgetter


# %%


class Data:
    """Base class to simulate or import data."""

    def __init__(self, batch_size, symbol=None):
        """Import data from dictionaries.

        If stock symbol is defined, individual stock dictionary is considered.
        Else, all stocks are used simultaneously in the analysis.
        Data dictionaries contain:
            - S0
            - dW
            - avg_bin_spread
            - avg_bin_volume

        Parameters
        ----------
        batch_size : TYPE
            Number of samples being generated.
        symbol : string, optional
            Stock ticker symbol. The default is None, and refers to using all
            stocks simultaneously.

        Returns
        -------
        None.

        """
        self.batch_size = batch_size

        if symbol:  # pick specific stock
            self.symbol = symbol

            # Import dict from pickled file
            filename = '../Data/Stock_{}/stock_data_dict'.format(self.symbol)
            infile = open(filename, 'rb')
            self.stock_data_dict = pickle.load(infile)
            infile.close()

        else:  # for all stocks at once

            # Import dict from pickled file
            filename = '../Data/all_stock_data_dict'
            infile = open(filename, 'rb')
            self.stock_data_dict = pickle.load(infile)
            infile.close()

    def get_X0(self, batch_size):
        """Initialize agent's initial wealth as zero.

        Parameters
        ----------
        batch_size : TYPE
            Number of samples being generated.

        Returns
        -------
        X0_train : TYPE
            DESCRIPTION.

        """
        X0_train = tf.zeros([batch_size, 1],
                            dtype=tf.float32)
        return X0_train

    def get_Q0(self, batch_size, avg_daily_volume):
        """Generate agent's initial inventory.

        Pick Q0_train to be uniform in [-.2,-.04]U[.04,.2]

        Parameters
        ----------
        batch_size : TYPE
            Number of samples being generated.
        avg_daily_volume : TYPE
            DESCRIPTION.

        Returns
        -------
        Q0_train : TYPE
            A fraction of the average daily volume of stock.

        """
        aux = np.where(tf.random.uniform([batch_size, 1], 0, 1) > .5, 1, -1)

        Q0_train = aux * tf.random.uniform([batch_size, 1],
                                           minval=.04,
                                           maxval=.2,
                                           dtype=tf.float32,
                                           seed=1)

        return Q0_train

    def get_index(self, batch_size):
        """Generate indices to sample pairs (S0, dW).

        Parameters
        ----------
        batch_size : TYPE
            Number of indices being generated.

        Returns
        -------
        idx : TYPE
            Choice of random indices to sample a matching pair (S0, dW).

        """
        # Prepare for resampling S0 and dW
        idx = np.random.choice(self.stock_data_dict['dW'][0].values.shape[0],
                               self.batch_size,
                               replace=True)
        return idx

    def _get_variable_bins(self, variable):
        """Average of volume or spread bins over all days for a certain stock/symbol.

        Parameters
        ----------
        variable : string
            Either 'avg_bin_volume' or 'avg_bin_spread'.

        Returns
        -------
        TYPE
            Average bin spread or volume, in tensor format.

        """
        avg_bin_variable = self.stock_data_dict[variable]
        avg_bin_variable = tf.convert_to_tensor(avg_bin_variable, dtype=np.float32)

        return avg_bin_variable  # shape = (1,78)

    def _get_flat_bins(self, binned_variable):
        """Get average of binned variables.

        Parameters
        ----------
        binned_variable : TYPE
            Either 'avg_bin_volume' or 'avg_bin_spread'.

        Returns
        -------
        TYPE
            Average over all values.

        """
        return tf.reduce_mean(binned_variable)

    def transform(self, param, avg_bin_spread, avg_bin_volume, dt):
        """Transform alpha/kappa according to avg_bin_spread / avg_bin_volume * dt.

        Takes into account the different spread and volume
        patterns throughout the trading day, whether in 'flat' or 'seasonality' mode.
        we divide by dt due to the way alpha and kappa were constructed, using 5 min bins.
        """
        new_param = param * (avg_bin_spread/10000) / avg_bin_volume / dt
        # Note: Here we are dividing by 10k because the avg_bin_spread
        # was erroneously in bps(*), and it should be in dollars.
        # (*) see ../../Python/FilePreprocessing/g_stock_data_dict.py for mistake origin

        return new_param


class Simulate(Data):
    """Simulate data using Monte Carlo."""

    def __init__(self, batch_size, N_T, dt, S0=100, symbol=None, seasonality="yes"):
        super(Simulate, self).__init__(batch_size, symbol)
        self.N_T = N_T
        self.dt = dt
        self.S0 = S0
        self.seasonality = seasonality

    def get_dW(self, batch_size, idx=0):
        """Generate mid-price increments."""
        dW = tf.random.normal([batch_size, self.N_T],
                              mean=0.0,
                              stddev=1.0,
                              dtype=tf.float32,
                              seed=1) * np.sqrt(self.dt)
        # needs to have same order of magnitude as real data -> std is same as stock std (bins)
        return dW

    def get_S0(self, batch_size, idx):
        """Generate initial price."""
        if self.S0:  # if user specifies S0, use it
            S0_train = self.S0 * tf.ones([self.batch_size, 1],
                                         dtype=tf.float32)

        else:  # get from stock_data_dict['S0']
            S0_train = list(itemgetter(*list(idx))(self.stock_data_dict['S0'])) # gets items according to self.idx
            S0_train = tf.convert_to_tensor(S0_train, dtype=np.float32)
            S0_train = tf.reshape(S0_train, [batch_size, 1])

        return S0_train

    def set_bins(self, binned_variable):
        avg_bin_variable = self._get_variable_bins(binned_variable)

        if self.seasonality == "no":  # for 'flat' mode
            avg_bin_variable = self._get_flat_bins(avg_bin_variable) \
                        * tf.ones([1, self.N_T], dtype=tf.float32)

        return avg_bin_variable


class Import(Data):
    """Import real TAQ data."""

    def __init__(self, batch_size, symbol=None, seasonality="yes"):
        super(Import, self).__init__(batch_size, symbol)
        self.seasonality = seasonality

    def get_dW(self, batch_size, idx):
        """Import mid-price increments."""
        dW = self.stock_data_dict['dW'][0].values[idx]
        dW = tf.convert_to_tensor(dW, dtype=np.float32)
        return dW

    def get_S0(self, batch_size, idx):
        """Import initial price."""
        S0_train = list(itemgetter(*list(idx))(self.stock_data_dict['S0']))  # gets items according to self.idx
        S0_train = tf.convert_to_tensor(S0_train, dtype=np.float32)
        S0_train = tf.reshape(S0_train, [batch_size, 1])

        return S0_train  # shape = (batch_size,1)

    def set_bins(self, binned_variable):
        avg_bin_variable = self._get_variable_bins(binned_variable)

        if self.seasonality == "no":  # for 'flat' mode
            self.N_T = self.stock_data_dict['dW'][0].shape[1]
            avg_bin_variable = self._get_flat_bins(avg_bin_variable) \
                * tf.ones([1, self.N_T], dtype=tf.float32)

        return avg_bin_variable


