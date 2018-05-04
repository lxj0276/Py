# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:57:35 2018

@author: s_zhangyw
"""

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


def max_entropy_2(Sigma):
    
    def safe_log(arr):
        brr = np.zeros_like(arr)
        brr = np.where(arr>0, np.log(arr), 0)
        return brr

    def func(x, Sigma, sign=1.0):
        Sigma = (Sigma + Sigma.T) / 2.0
        eig_values, eig_vectors= np.linalg.eig(Sigma)
        y = eig_vectors.T.dot(x)
        v = (y**2) * eig_values
        p = v / v.sum()
        res = (p * safe_log(p))[2:].sum()
        return res * sign
    
    return optimize(func, Sigma)





