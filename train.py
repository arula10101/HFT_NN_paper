#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:17:26 2019

@author: laura
"""


import time
import numpy as np
import tensorflow as tf
import datetime
import os
from copy import copy
from distutils.dir_util import copy_tree

from data import Simulate, Import
from get_config import hexConfig

# %%


class Solver:
    """NN trainer."""

    def __init__(self, model, config):
        """Initialization based on model chosen and configurations."""
        self.config = config

        # define attributes from config file
        for (attr, value) in self.config.items():
            setattr(self, attr, value)
        self.dt = self.T / self.N_T

        if not self.phi:
            self.phi = tf.random.uniform(shape=[1, 1],
                                         minval=0.007,
                                         maxval=0.7,
                                         dtype=tf.dtypes.float32)[0, 0]

        if not self.A:
            self.A = tf.random.uniform(shape=[1, 1],
                                       minval=1,
                                       maxval=10,
                                       dtype=tf.dtypes.float32)[0, 0]

        # define control for the model
        self.nu_theta = model

    def _create_checkpoint(self, optimizer, configs):
        # CREATE CHECKPOINT FOR TRANSFER LEARNING
        # saves weights and biases for the model

        # When importing data, copy checkpoints from simulation into new
        # directory to do transfer learning. If simulation directory does
        # not exist, then creates new one. Note: it will only be copied
        # if simulation is done first.

        config_ = copy(configs)
        hc = hexConfig()
        hex_conf = hc.hex_config(config_)
        if config_["simulate"] == "yes":
            checkpoint_directory = "checkpoints/checkpoint_{}".format(hex_conf)
        else:
            to_dir = "checkpoints/checkpoint_{}".format(hex_conf)
            config_["simulate"] = "yes"
            from_hex = hc.hex_config(config_)
            from_dir = "checkpoints/checkpoint_{}".format(from_hex)
            if os.path.isdir(from_dir) and not os.path.isdir(to_dir):
                copy_tree(from_dir, to_dir)
            checkpoint_directory = to_dir

        # create checkpoint
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self.nu_theta,
                                         optimizer_step=tf.train.get_or_create_global_step())
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
        return checkpoint, checkpoint_prefix

    def get_data(self, stock_data, batch_size, tile_size, avg_bin_volume):
        idx = stock_data.get_index(batch_size)
        S0_train = stock_data.get_S0(batch_size, idx)
        dW = stock_data.get_dW(batch_size, idx)
        X0_train = stock_data.get_X0(batch_size)
        Q0_train = stock_data.get_Q0(batch_size * tile_size, avg_bin_volume)

        # Use different Q0 to train with the same pair S0-dW (increase number of samples)
        S0_train = tf.tile(S0_train, [tile_size, 1])
        dW = tf.tile(dW, [tile_size, 1])
        X0_train = tf.tile(X0_train, [tile_size, 1])
        return X0_train, S0_train, Q0_train, dW

    def update_states(self, X1, X2, X, S, Q, nu_t, dW, i_t):
        # use model dynamics to update state variables given the control at time i_t
        # X = X - nu_t * (S + self.kappa[0][i_t-1] * nu_t) * self.dt
        X1 = X1 - nu_t * S * self.dt
        X2 = X2 - nu_t * self.kappa[0][i_t-1] * nu_t * self.dt
        X = X1 + X2
        # X = X - nu_t * S  * self.dt
        # a = - nu_t * (S + self.kappa[0][i_t-1] * nu_t) * self.dt
#        print('x',a[0][0])
        S = S + self.alpha[0][i_t-1] * nu_t * self.dt + self.sigma * tf.slice(dW, [0, i_t-1], [-1, 1])
        # b = self.alpha[0][i_t-1] * nu_t * self.dt
        # print("b",b[0][0])
        # c = self.sigma * tf.slice(dW,[0,i_t-1],[-1,1])
        # print("c",c[0][0])
        Q = Q + nu_t * self.dt
        # c=nu_t * self.dt
        # print(c[0][0])
        return X1, X2, X, S, Q

    def _create_stack(self, variable, n_samples):
        variable_stack = tf.reshape(tf.tile([tf.cast(variable, tf.float32)],
                                    tf.stack([n_samples])),
                                    [n_samples, 1])
        return variable_stack  # same size as X

    # ========== BUILDING === describes the computation of the loss function
    def fwd_pass(self, X0_train, S0_train, Q0_train, dW, avg_bin_spread, avg_bin_volume):
        # INITIALIZATION
        self.N_T = dW.shape[1].value  # number of time steps
        self.dt = self.T / self.N_T
        n_samples = tf.shape(X0_train)[0]  # nb of samples in one mini-batch
        avg_daily_volume = tf.reduce_sum(avg_bin_volume)
        self.loss = tf.constant([0], dtype=tf.float32, shape=[1])

        # PREPARE FIRST STATE AND CONTROL
        i_t = 0  # index of time step
        X = X0_train  # initial position for wealth
        X1 = X0_train/2
        X2 = X0_train/2
        S = S0_train  # initial position for price
        Q = Q0_train  # initial position for inventory, as % of daily volume

        t_stack = self._create_stack(self.T - i_t*self.dt, n_samples)
#        spread_stack = self._create_stack(avg_bin_spread[0][i_t], n_samples)
#        volume_stack = self._create_stack(avg_bin_volume[0][i_t], n_samples)
        #A_stack = self._create_stack(self.A, n_samples)
        #phi_stack = self._create_stack(self.phi, n_samples)

        # inputs = tf.concat([t_stack, Q, phi_stack, A_stack], axis=1)
        inputs = tf.concat([t_stack, Q], axis=1)
        # if changing the inputs, check CJ_HJBSolver in models.py
        if self.config['model'] in ("HJB", "HJB_q32"):
            nu_t = self.nu_theta.call(i_t,
                                      self.dt,
                                      self.T,
                                      self.N_T,
                                      self.A,
                                      self.phi,
                                      self.alpha[0][i_t],
                                      self.kappa[0][i_t],
                                      inputs)
        else:
            nu_t = self.nu_theta.call(inputs)

        nu_t = nu_t * avg_daily_volume
        Q = Q * avg_daily_volume

        self.X_path = X  # keep track of position's path for wealth for t=0
        self.X1_path = X1  # keep track of position's path for wealth for t=0
        self.X2_path = X2  # keep track of position's path for wealth for t=0

        self.S_path = S  # keep track of position's path for price
        self.Q_path = Q  # keep track of position's path for inventory
        self.control_path = nu_t  # keep track of control's path

        # LOOP IN TIME
        for i_t in range(1, self.N_T):
            # self.N_T+1 -> by removing the +1, we are no longer calculating
            # the unnecessary nu_t
            # hence, we will no longer compare the NN result to the last step
            # in the HJB control
            # but also, we no longer calculate the last step of the HJB control
            nu_t = tf.reshape(nu_t, shape=[n_samples, 1])
            #self.loss += self.phi * tf.reduce_mean(tf.math.pow(tf.math.abs(Q), 3/2)) * self.dt
            self.loss += self.phi * tf.reduce_mean(tf.square(Q)) * self.dt  # update loss

            # UPDATE STATE VARIABLES
            X1, X2, X, S, Q = self.update_states(X1, X2, X, S, Q, nu_t, dW, i_t)
            tf.debugging.assert_greater(S, 0.0)

            t_stack = self._create_stack(self.T - i_t*self.dt, n_samples)
#            spread_stack = self._create_stack(avg_bin_spread[0][i_t], n_samples)
#            volume_stack = self._create_stack(avg_bin_volume[0][i_t], n_samples)

            # A_stack = self._create_stack(self.A, n_samples)
            # phi_stack = self._create_stack(self.phi, n_samples)

            Q = Q / avg_daily_volume
            # inputs = tf.concat([t_stack, Q, phi_stack, A_stack], axis=1)
            inputs = tf.concat([t_stack, Q], axis=1)

            if (self.config['model'] == 'HJB' or self.config['model'] == 'HJB_q32'):
                nu_t = self.nu_theta.call(i_t,
                                          self.dt,
                                          self.T,
                                          self.N_T,
                                          self.A,
                                          self.phi,
                                          self.alpha[0][i_t],
                                          self.kappa[0][i_t],
                                          inputs)
            else:
                nu_t = self.nu_theta.call(inputs)
            nu_t = nu_t * avg_daily_volume
            Q = Q * avg_daily_volume

            self.X_path = tf.concat([self.X_path, X], axis=1)
            self.X1_path = tf.concat([self.X1_path, X1], axis=1)
            self.X2_path = tf.concat([self.X2_path, X2], axis=1)

            self.S_path = tf.concat([self.S_path, S], axis=1)
            self.Q_path = tf.concat([self.Q_path, Q], axis=1)
            self.control_path = tf.concat([self.control_path, nu_t], axis=1)

        # Update and save for last step
        X1, X2, X, S, Q = self.update_states(X1, X2, X, S, Q, nu_t, dW, i_t=self.N_T)

        self.X_path = tf.concat([self.X_path, X], axis=1)
        self.X1_path = tf.concat([self.X1_path, X1], axis=1)
        self.X2_path = tf.concat([self.X2_path, X2], axis=1)

        self.S_path = tf.concat([self.S_path, S], axis=1)
        self.Q_path = tf.concat([self.Q_path, Q], axis=1)
        self.control_path = tf.concat([self.control_path, nu_t], axis=1)

        # COST FOR LAST STEP
        #self.loss -= tf.reduce_mean(X + Q * S - self.A * tf.math.pow(tf.math.abs(Q), 3/2))
        self.loss -= tf.reduce_mean(X + Q * (S - self.A * Q))

        tf.contrib.summary.scalar('X_T', tf.reduce_mean(X))

    def control(self):
        """TODO.

        Returns
        -------
        None.

        """
        # Generate data:
        if self.simulate == 'yes':
            stock_data = Simulate(self.minibatch_train_size,
                                  self.N_T,
                                  self.dt,
                                  self.S0,
                                  self.stock_symbol,
                                  self.seasonality)
        else:
            stock_data = Import(self.minibatch_train_size,
                                self.stock_symbol,
                                self.seasonality)

        self.avg_bin_spread = stock_data.set_bins('avg_bin_spread')
        self.avg_bin_volume = stock_data.set_bins('avg_bin_volume')

        self.alpha = stock_data.transform(self.alpha,
                                          self.avg_bin_spread,
                                          self.avg_bin_volume,
                                          self.dt)
        self.kappa = stock_data.transform(self.kappa,
                                          self.avg_bin_spread,
                                          self.avg_bin_volume,
                                          self.dt)

        X0_train, S0_train, Q0_train, dW = self.get_data(stock_data,
                                                         self.minibatch_train_size,
                                                         self.tile_size,
                                                         self.avg_bin_volume)

        self.fwd_pass(X0_train,
                      S0_train,
                      Q0_train,
                      dW,
                      self.avg_bin_spread,
                      self.avg_bin_volume)

        curr_loss = self.loss
        curr_wealth = tf.reduce_mean(self.X_path[:, -1])

        print("Control Loss: {}".format(curr_loss.numpy()[0]))
        print("Control Wealth: {}".format(curr_wealth.numpy()))


#    @tf.function -> for version 2.0 -> will compile in c++ the second time it runs,
#     so it will be faster
    def train(self, n_iterSGD):
        """TODO
        

        Parameters
        ----------
        n_iterSGD : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # TRAIN THE NN
        print('========== START TRAINING ==========')

        start_time = time.time()

        # Create summary logs for the loss function
        tf.reset_default_graph()
        log_dir="summary_logs/" + datetime.datetime.now().strftime("%Y:%m:%d-%H.%M")
        summary_writer = tf.contrib.summary.create_file_writer(log_dir,
                                                               flush_millis=10000)
        
        # Define optimizer with learning rate of SGD
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Create checkpoints
        checkpoint, checkpoint_prefix = self._create_checkpoint(optimizer,
                                                                self.config)

        # Generate data:
        if self.simulate == 'yes':
            stock_data = Simulate(self.minibatch_train_size,
                                  self.N_T,
                                  self.dt,
                                  self.S0,
                                  self.stock_symbol,
                                  self.seasonality)
        else:
            stock_data = Import(self.minibatch_train_size,
                                self.stock_symbol,
                                self.seasonality)

        self.avg_bin_spread = stock_data.set_bins('avg_bin_spread')
        self.avg_bin_volume = stock_data.set_bins('avg_bin_volume')

        self.alpha = stock_data.transform(self.alpha,
                                          self.avg_bin_spread,
                                          self.avg_bin_volume,
                                          self.dt)
        self.kappa = stock_data.transform(self.kappa,
                                          self.avg_bin_spread,
                                          self.avg_bin_volume,
                                          self.dt)
        phi_bound = max(max(self.alpha**2/self.kappa))

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():

            # Loop on iterations of SGD
            for i_iterSGD in range(n_iterSGD):
                # AT EACH ITERATION: make a forward run to compute the gradient
                # of the loss and update the NN

                X0_train, S0_train, Q0_train, dW = self.get_data(stock_data,
                                                                 self.minibatch_train_size,
                                                                 self.tile_size,
                                                                 self.avg_bin_volume)

                with tf.GradientTape() as tape:
                    # use tape to compute the gradient; see below
                    self.fwd_pass(X0_train,
                                  S0_train,
                                  Q0_train,
                                  dW,
                                  self.avg_bin_spread,
                                  self.avg_bin_volume)
                    curr_loss = self.loss

                # Compute gradient w.r.t. parameters of NN:
                grads = tape.gradient(curr_loss, self.nu_theta.variables)
                # Make one SGD step:
                optimizer.apply_gradients(zip(grads, self.nu_theta.variables),
                                          global_step=tf.train.get_or_create_global_step())

                # Print and save:
                if i_iterSGD % self.n_displaystep == 0:
                    print("** cumulated runtime: %4u" % (time.time()-start_time))
                    self.validation(i_iterSGD, self.config)


#                # SAVE LOSS FUNCTION TO DISPLAY ON TENSORBOARD
#                tf.contrib.summary.scalar("loss", curr_loss)

                # SAVE CHECKPOINT PERIODICALLY
                if i_iterSGD % self.n_checkpoint == 0:
                    checkpoint.save(file_prefix=checkpoint_prefix)

                # Change to a new phi for training on different values
                if self.change_risk_params and i_iterSGD % self.change_risk_params == 0:
                    self.phi = tf.random.uniform(shape=[1, 1],
                                                 minval=phi_bound,
                                                 maxval=100*phi_bound,
                                                 dtype=tf.dtypes.float32)[0, 0]
                    curr_phi = self.phi
                    tf.contrib.summary.scalar("phi", curr_phi)

                    self.A = tf.random.uniform(shape=[1, 1],
                                               minval=1,
                                               maxval=10,
                                               dtype=tf.dtypes.float32)[0, 0]
                    
                    curr_A = self.A
                    tf.contrib.summary.scalar("A", curr_A)


        end_time = time.time()
        print('========== END TRAINING ==========')
        print("running time for training: %.3f s" % (end_time - start_time))

    def validation(self, i_iterSGD, config_):
        # CHECK THE GENERALIZATION ERROR
        # (aggregate all samples so far, instead of just mini-batch error)

        if self.simulate == 'yes':
            stock_data = Simulate(self.validation_size,
                                  self.N_T,
                                  self.dt,
                                  self.S0,
                                  self.stock_symbol,
                                  self.seasonality)
        else:
            stock_data = Import(self.validation_size,
                                self.stock_symbol,
                                self.seasonality)

        X0_valid, S0_valid, Q0_valid, dW = self.get_data(stock_data,
                                                         self.validation_size,
                                                         self.tile_size,
                                                         self.avg_bin_volume)
        self.fwd_pass(X0_valid,
                      S0_valid,
                      Q0_valid,
                      dW,
                      self.avg_bin_spread,
                      self.avg_bin_volume)

        curr_loss = self.loss
        curr_wealth = tf.reduce_mean(self.X_path[:, -1])

        # SAVE LOSS FUNCTION TO DISPLAY ON TENSORBOARD
        tf.contrib.summary.scalar("loss", curr_loss)

        print("Loss at step {}: {}".format(i_iterSGD, curr_loss.numpy()[0]))
        print("Wealth at step {}: {}".format(i_iterSGD, curr_wealth.numpy()))
        hex_conf = hexConfig().hex_config(self.config)
        datafile_name = "logs/log_{}/data_solution_iterSGD{}.npz".format(hex_conf, i_iterSGD)
        np.savez(datafile_name,
                 X_path=self.X_path,
                 S_path=self.S_path,
                 Q_path=self.Q_path,
                 Control_path=self.control_path,
                 alpha=self.alpha,
                 kappa=self.kappa)
