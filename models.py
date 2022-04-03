#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:22:36 2019

@author: laura
"""

import numpy as np

# %%

import tensorflow as tf


class NNSolver(tf.keras.Model):
    
    def __init__(self):
        # build the NN; the input size will be determined automatically when it is called for the first time
        super(tf.keras.Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=5, 
                                            activation='tanh', 
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2), 
                                            bias_initializer='zeros')
        self.drop1 = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(units=5,
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2), 
                                            bias_initializer='zeros')
        self.drop2 = tf.keras.layers.Dropout(rate=0.2)
        self.dense3 = tf.keras.layers.Dense(units=5,
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, 
                                                                                                  stddev=0.2),
                                            bias_initializer='zeros')
        self.dense4 = tf.keras.layers.Dense(units=1, 
                                            # activation='tanh',
                                            # activation = 'sigmoid',
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, 
                                                                                                  stddev=0.2),
                                            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, 
                                                                                                stddev=0.2)) 

    def call(self, input):
        # applies the NN (composition of layers) on the input and returns the output
        result = self.dense1(input)
        result = self.drop1(result)
        result = self.dense2(result)
        result = self.drop2(result)
        result = self.dense3(result)
        result = self.dense4(result)
        return result


# equivalent to controller
class CJ_HJBSolver:
    """Solves HJB equation.

    h2 is solved backwards using Riccati equation in Carmona, Delarue (2018).
    h1 is solved forward by ODE, given the solution to h2.

    """

    def __init__(self):
        """__init___ method initializes HJB solver class using 'pass'."""
        pass

    def call(self, i_t, dt, T, N_T, A, phi, alpha_t, kappa_t, input):
        """TODO."""
        self.i_t = i_t
        self.dt = dt
        self.T = T
        self.N_T = N_T
        self.A = A
        self.phi = phi
        self.alpha_t = alpha_t
        self.kappa_t = kappa_t

        h2 = self._get_h2(alpha_t, kappa_t)
        h1 = self._get_h1(h2, alpha_t, kappa_t)

#        t = input[:,0] # time to maturity, in [0,T]
        Q_t = input[:, 1]

        h2_t = h2[self.i_t]
        h1_t = h1[self.i_t]

        # calculate speed of trading nu_t
        nu_t = (h1_t + (alpha_t + h2_t) * Q_t) / (2 * kappa_t)
        nu_t = tf.reshape(nu_t, (nu_t.shape[0], 1))

        return nu_t

    def _get_h2(self, alpha_t, kappa_t):
        # Solves h2 backwards using closed form solution for Riccati equation
        # (see eq 2.50 in Carmona, Delarue (2018))
        # Auxiliary variables for the Riccati equation (see eq 2.50 in Carmona, Delarue (2018))
        # Define A, B, and C based on your own Riccati eq
        A_ric = - alpha_t/(2*kappa_t)  # this is not the same as A used in the NN!!
        B_ric = -1/(2*kappa_t)
        C_ric = np.square(alpha_t)/(2*kappa_t) - 2*self.phi

        R = A_ric**2 + B_ric * C_ric
        assert R > 0
#        print(np.square(alpha_t)/(2*kappa_t), 2*self.phi  )
#        print(A_ric**2, B_ric, C_ric)
        delta_minus = - A_ric - np.sqrt(R)
        delta_plus = - A_ric + np.sqrt(R)
        h2_T = -2*self.A
        gamma = h2_T
        assert gamma*B_ric > 0
        assert B_ric * C_ric > 0

        h2_rev = []
        for i in range(self.N_T, 0, -1):
            j = i * self.dt
#            print(delta_plus - delta_minus)
#            print(self.T-j)
            expon = np.exp((delta_plus - delta_minus)*(self.T-j))  # note: self.T-j = input t
            num = - C_ric * (expon - 1) - gamma * (delta_plus*expon - delta_minus)
            den = (delta_minus * expon - delta_plus) - gamma * B_ric * (expon - 1)
            h2_rev_t = num/den
            h2_rev.append(h2_rev_t)
        self.h2 = h2_rev[::-1]
        return self.h2

    def _get_h1(self, h2, alpha_t, kappa_t):
        # Solves for h1 forward using h2

        def u():
            u_t = np.zeros(self.N_T)
            for i in range(1, self.N_T+1):
                u_t[i-1] = np.exp(np.sum(np.divide(np.add(h2[:i],
                                                          alpha_t),
                                                   2.*kappa_t)*self.dt))
            return u_t

        def integ(u_t):
            mu_t = 0  # for now, we are ignoring the mean field term!! (hence, h1 will be 0.)
            integ = np.cumsum(u_t * mu_t * self.dt)
            return integ

        u_ = u()
        integ_ = integ(u_)

        # h1_const = (h1(T) * u(T)) + alpha * int_0ˆT u(s) ds) , but notice h1(T) = 0
        h1_const = alpha_t * integ_[-1]

        self.h1 = []
        for i in range(self.N_T):
            self.h1.append((1/u_[i]) * (-alpha_t * integ_[i] + h1_const))

        return self.h1


class CJ_HJBSolver_q32:
    """Solves HJB equation for case gamma=3/2.

    h4 is solved backwards using Riccati equation in Carmona, Delarue (2018).
    h3 and h2 are solved forward by ODE, given the solution to h4 and (h4,h3), respectively.

    """

    def __init__(self):
        """__init___ method initializes HJB solver class using 'pass'."""
        pass

    def call(self, i_t, dt, T, N_T, A, phi, alpha_t, kappa_t, input):
        """TODO."""
        self.i_t = i_t
        self.dt = dt
        self.T = T
        self.N_T = N_T
        self.A = A
        self.phi = phi
        self.alpha_t = alpha_t
        self.kappa_t = kappa_t

        h4 = self._get_h4(alpha_t, kappa_t)
        h3 = self._get_h3(h4, alpha_t, kappa_t)
        h2 = self._get_h2(h3, h4, alpha_t, kappa_t)

#        t = input[:,0] # time to maturity, in [0,T]
        Q_t = input[:, 1]

        h4_t = h4[self.i_t]
        h3_t = h3[self.i_t]
        h2_t = h2[self.i_t]

        # calculate speed of trading nu_t
        nu_t = (h2_t + (alpha_t + 2*h4_t) * Q_t + (3/2)*h3_t*(Q_t**(3/2))) / (2 * kappa_t)
        nu_t = tf.reshape(nu_t, (nu_t.shape[0], 1))

        return nu_t

    def _get_h4(self, alpha_t, kappa_t):
        # Solves h4 backwards using closed form solution for Riccati equation
        # (see eq 2.50 in Carmona, Delarue (2018))
        # Auxiliary variables for the Riccati equation (see eq 2.50 in Carmona, Delarue (2018))
        # Define A, B, and C based on your own Riccati eq
        A_ric = - alpha_t/(2*kappa_t)  # this is not the same as A used in the NN!!
        B_ric = -1/kappa_t
        C_ric = np.square(alpha_t)/(4*kappa_t)

        R = A_ric**2 + B_ric * C_ric
        # assert R > 0
#        print(np.square(alpha_t)/(2*kappa_t), 2*self.phi  )
#        print(A_ric**2, B_ric, C_ric)
        delta_minus = - A_ric - np.sqrt(R)
        delta_plus = - A_ric + np.sqrt(R)
        h4_T = -self.A
        gamma = h4_T
        assert gamma*B_ric > 0
        # assert B_ric * C_ric > 0

        h4_rev = []
        for i in range(self.N_T, 0, -1):
            j = i * self.dt
            expon = np.exp((delta_plus - delta_minus)*(self.T-j))  # note: self.T-j = input t
            num = - C_ric * (expon - 1) - gamma * (delta_plus*expon - delta_minus)
            den = (delta_minus * expon - delta_plus) - gamma * B_ric * (expon - 1)
            h4_rev_t = num/den
            h4_rev.append(h4_rev_t)
        self.h4 = h4_rev[::-1]
        return self.h4
    
    def _get_h3(self, h4, alpha_t, kappa_t):
        # Solves for h3 forward using h4
        
        def u():
            u_t = np.zeros(self.N_T)
            for i in range(1, self.N_T+1):
                u_t[i-1] = np.exp(3*np.sum(np.divide(np.add(2*h4[:i],
                                                          alpha_t),
                                                   4.*kappa_t)*self.dt))
            return u_t
        
        def integ(u_t):
            integ = np.cumsum(u_t * self.phi * self.dt)
            return integ
            
        u_ = u()
        integ_ = integ(u_)

        # h3_const = (h3(T) * u(T)) + phi * int_0ˆT u(s) ds) , but notice h3(T) = -A
        h3_const = -self.A * u_[-1] + self.phi * integ_[-1]

        self.h3 = []
        for i in range(self.N_T):
            self.h3.append((1/u_[i]) * (integ_[i] + h3_const))

        return self.h3

    def _get_h2(self, h3, h4, alpha_t, kappa_t):
        
        mu = 0

        def u():
            u_t = np.zeros(self.N_T)
            for i in range(1, self.N_T+1):
                u_t[i-1] = np.exp(np.sum(np.divide(np.add(2*h4[:i],
                                                          alpha_t),
                                                   2.*kappa_t)*self.dt))
            return u_t

        def integ(u_t):
            integ = np.cumsum(u_t * (-self.alpha_t * mu - (9/16) * np.square(h3)) * self.dt)
            return integ

        u_ = u()
        integ_ = integ(u_)

        # h2_const = h2(T) * u(T)) + int_0ˆT u(s) q(s) ds , but notice h2(T) = 0
        h2_const = -integ_[-1]

        self.h2 = []
        for i in range(self.N_T):
            self.h2.append((1/u_[i]) * (integ_[i] + h2_const))
  
        return self.h2
        
    # h1 and h0 are not used in the control, thus are not coded. 
        


# deeper = more detail in the pattern
# wider = more types of pattern
        

    