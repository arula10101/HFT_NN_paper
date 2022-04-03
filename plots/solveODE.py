#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 20:21:04 2020

@author: Laura Simonsen Leal
"""

import numpy as np

# %%

class solveODEsystem:
    """
    Class to solve the ODE system specific to the problem.
        1) h2 is solved by Riccati equation
        2) h1 is solved by a non-homogeneous first order ODE
        3) h0 is solved by a homogeneous first order ODE

    """

    def __init__(self, kappa=0.24, alpha=0.16, phi=0.7, A=10, N_T=78):

        # Coefficients of the problem
        self.T = 1.0  # time horizon
        self.N_T = N_T  # number of time steps
        self.dt = self.T/self.N_T
        self.kappa = kappa
        self.phi = phi
        self.alpha = alpha
        self.A = A
        self.sigma = 0.2

        # terminal conditions
        # come from setting V(T,q) = X(T) + Q(T)*S(T) + v(T,Q(T))=
        #                          = X(T) + Q(T)*S(T) + (h0(T)+h1(T)*q '+' 1/2 *h2(T)*qˆ2)
        # equal to the terminal cost X(T) + Q(T)*S(T) - A*Q(T)ˆ2
        # Hence for h2(T): 1/2*h2(T) = -A   => h2(T) = -2A
        self.h0_T = 0
        self.h1_T = 0
        self.h2_T = -2*self.A

        # Define h0, h1, h2
        self.h0 = None
        self.h1 = None
        self.h2 = None


    def solveRiccati(self):
        # Solves h2 backwards using closed form solution for Riccati equation
        # (see eq 2.50 in Carmona, Delarue (2018))
        # Auxiliary variables for the Riccati equation (see eq 2.50 in Carmona, Delarue (2018))
        # Define A, B, and C based on your own Riccati eq
        A_ric = - self.alpha/(2*self.kappa)  # this is not the same as A used in the NN!!
        B_ric = -1/(2*self.kappa)
        C_ric = np.square(self.alpha)/(2*self.kappa) - 2*self.phi

        R = A_ric**2 + B_ric * C_ric
        assert R > 0
        delta_minus = - A_ric - np.sqrt(R)
        delta_plus = - A_ric + np.sqrt(R)
        gamma = self.h2_T
        assert gamma*B_ric > 0
        assert B_ric * C_ric > 0

        h2_rev = []
        for i in range(self.N_T, 0, -1):
            t = i * self.dt
            expon = np.exp((delta_plus - delta_minus)*(self.T-t))
            num = - C_ric * (expon - 1) - gamma * (delta_plus*expon - delta_minus)
            den = (delta_minus * expon - delta_plus) - gamma * B_ric * (expon - 1)
            h2_rev_t = num/den
            h2_rev.append(h2_rev_t)
        self.h2 = h2_rev[::-1]
        return self.h2


    def solveH1(self):
        # Solves for h1 forward using h2

        def u():
            u_t = np.zeros(self.N_T)
            for i in range(1, self.N_T+1):
                u_t[i-1] = np.exp(np.sum(np.divide(np.add(self.h2[:i], self.alpha),
                                                   2.*self.kappa)*self.dt))
            return u_t

        def integ(u_t):
            mu_t = 0  # for now, we are ignoring the mean field term!!
            integ = np.cumsum(u_t * mu_t * self.dt)
            return integ

        u_ = u()
        integ_ = integ(u_)

        # h1_const = (h1(T) * u(T)) + alpha * int_0ˆT u(s) ds) , but notice h1(T) = 0
        h1_const = self.alpha * integ_[-1]

        self.h1 = []
        for i in range(self.N_T):
            self.h1.append((1/u_[i]) * (-self.alpha * integ_[i] + h1_const))

        return self.h1


    def solveH0(self):
        # h0_const = h0_t + (1/4kappa) * int_0^T h1^2 dt, and notice h0T = 0
        h0_const = np.divide(np.sum(np.square(self.h1) * self.dt), 4*self.kappa)

        self.h0 = []
        for i in range(self.N_T):
            self.h0.append(np.sum(-np.square(self.h1[:i+1])*self.dt)/(4*self.kappa)+h0_const)
        return self.h0

    def get_hs(self, Q):
        a = Q.shape[0]  # time steps

        h2 = self.solveRiccati()
        h2 = np.array(h2).reshape((a, 1))

        h1 = self.solveH1()
        h1 = np.array(h1).reshape((a, 1))

        h0 = self.solveH0()
        h0 = np.array(h0).reshape((a, 1))

        return h2, h1, h0

    def u(self, Q):
        # u(t,q) = h2*q^2/2 + h1*q + h0, this is the ansatz, not the value function itself
        h2, h1, h0 = self.get_hs(Q)

        u_ = h2*np.square(Q)/2 + h1*Q + h0

        return u_

    def control(self, Q):

        h2, h1, _ = self.get_hs(Q)

        # ctrl = h2*Q + h1
        ctrl = (self.alpha*Q + (h1+h2*Q)) / (2*self.kappa)

        return ctrl

    def control_dyn(self, Q, S, gamma):  # get full matrices
        self.h2, self.h1, _ = self.get_hs(Q.T)
        n_samples = len(Q)
        self.Q = np.zeros([n_samples, self.N_T])
        self.Q[:, 0] = Q[:, 0]
        self.X = np.zeros([n_samples, self.N_T])
        self.S = np.zeros([n_samples, self.N_T])
        self.S[:, 0] = S[:, 0]

        nu_t = np.zeros((n_samples, self.N_T))

        for i in range(self.N_T-1):
            # MOVE FORWARD
            nu_t[:, i] = (1/(2*self.kappa)) * ((self.alpha +
                                                self.h2[i]) * self.Q[:, i] +
                                               self.h1[i])
            dW = np.random.normal(0.0, 1.0, n_samples) * np.sqrt(self.dt)
            self.S[:, i+1] = self.S[:, i] + (self.alpha * nu_t[:, i] * self.dt) + \
                self.sigma * dW
            self.Q[:, i+1] = self.Q[:, i] + nu_t[:, i] * self.dt
            self.X[:, i+1] = self.X[:, i] - \
                nu_t[:, i] * (self.S[:, i] + self.kappa * nu_t[:, i]) * self.dt

        nu_t[:, -1] = (self.alpha * self.Q[:, -1] +
                       self.h1[-1] + self.h2[-1] * self.Q[:, -1])/(2*self.kappa)

        return nu_t, self.Q, self.X, self.S

    def valfct_dyn(self, Q, S, X):  # need the full matrices
        gamma = 2  # only
        # _, self.Q, self.X, self.S = self.control_dyn(Q0, S0, gamma)
        value_fct_dyn = - self.phi * np.sum(np.square(Q[:, :-1])*self.dt, axis=1) + \
                        X[:, -1] + self.Q[:, -1] * self.S[:, -1] - \
                        self.A * np.power(self.Q[:, -1], gamma)
        value_fct_dyn_mean = np.nanmean(value_fct_dyn)
        return value_fct_dyn, value_fct_dyn_mean

    def valfct_hs(self, Q, S, X):  # get full matrices, but need only initial values
        # gamma = 2 only
        h2, h1, h0 = self.get_hs(Q)
        value_fct_hs = X[:, 0]  \
                        + S[:, 0] * Q[:, 0] \
                        + h2[0]*np.square(Q[:, 0])/2 \
                        + h1[0]*Q[:, 0] \
                        + h0[0]
                        
    #    value_fct_hs_mean = np.mean(value_fct_hs)
        return value_fct_hs

