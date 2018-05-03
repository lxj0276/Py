# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:20:53 2018

@author: s_zhangyw
"""


import numpy as np
from scipy.optimize import minimize


def optimize(func, *args):

    N = len(args[0])
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})
    res = minimize(func, [1/N]*N, args=args, constraints=cons, bounds=[(0, 1)] * N, 
        method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


def max_sharp(r, Sigma):

    def func(x, r, Sigma, sign=-1.0):
        res = x.dot(r) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, r, Sigma)


def min_vol(Sigma):

    def func(x, Sigma, sign=1.0):
        res = (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, Sigma)


def vol_parity(sigma):

    return (1.0 / sigma) / (1.0 / sigma).sum()


def risk_parity(Sigma):

    def func(x, Sigma, sign=1.0):
        RC = x * Sigma.dot(x) / (x.dot(Sigma).dot(x)) ** 0.5
        res = ((np.tile(RC, (len(RC), 1)).T - RC) ** 2).sum().sum()
        return res * sign

    return optimize(func, Sigma)


def most_diversified(sigma, Sigma):

    def func(x, sigma, Sigma, sign=-1.0):
        res = x.dot(sigma) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, sigma, Sigma)


def most_decorr(Rho):
    def func(x, Rho, sign=1.0):
        res = x.dot(Rho).dot(x)
        return res * sign

    return optimize(func, Rho)

def max_entropy(Sigma):
    
    def safe_log(arr):
        brr = np.zeros_like(arr)
        brr = np.where(arr>0, np.log(arr), 0)
        return brr

    def func(x, Sigma, sign=1.0):
        eig_values, eig_vectors= np.linalg.eig(Sigma)
        y = eig_vectors.T.dot(x)
        v = (y**2) * (eig_values**2)
        p = v / v.sum()
        res = (p * safe_log(p)).sum().real
        return res * sign
    
    return optimize(func, Sigma)






