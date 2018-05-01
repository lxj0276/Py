# -*- encoding: utf-8 -*-
'''
投资组合函数：均值方差，风险平价，风险预算
'''

import numpy as np
from scipy.optimize import minimize

def mv(u, sigma, lmd=2.5):
    '''
    均值方差模型：输入均值向量，协方差矩阵和风险厌恶系数
    '''

    def func(x, u, sigma, lmd, sign=-1.0):
        N = len(sigma)
        L = 0
        for i in range(N):
            L = L + x[i] * u[i]
        R = 0
        for j in range(N):
            for k in range(N):
                R = R + x[j] * x[k] * sigma[j][k]

        return sign * (L - lmd / 2.0 * R)

    def func_deriv(x, u, sigma, lmd, sign=-1.0):
        N = len(sigma)
        dfdx = []
        for i in range(N):
            R = 0
            for j in range(N):
                R = R + sigma[i, j] * x[j]

            dfdxi = sign * (u[i] - lmd * R)
            dfdx.append(dfdxi)
        return np.array(dfdx)

    N = len(sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    # res = minimize(func, [1/N]*N, args=(u, sigma, lmd), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(u, sigma, lmd),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})
    return res.x


def rp(sigma):
    '''
    风险平价模型：输入协方差矩阵
    '''

    def func(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / R.dot(x)
        return sign * O

    def func_deriv(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x)
        U = R.dot(x)

        def drv(i, k, r=R, s=sigma, w=x):
            if i != k:
                return s[i][k] * w[i]
            else:
                return r.sum() + r[i]

        def drv_sigma(k, s=sigma, w=x):
            N = len(sigma)
            D = 0
            for i in range(N):
                D = D + s[k, j] * w[j]
            return D

        def drv_total(i, k):
            return (drv(i, k) * U ** 0.5 - (drv_sigma(k) * x[i] * R[i]) / (2 * U ** 0.5)) / U

        dfdx = []
        for k in range(N):
            S = 0
            for i in range(N):
                for j in range(N):
                    S = S + 2 * ((x[i] * R[i] - x[j] * R[j]) / R.dot(x)) \
                        * (drv_total(i, k) - drv_total(j, k))
            dfdxk = sign * S
            dfdx.append(dfdxk)

        return np.array(dfdx)

    N = len(sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    # res = minimize(func, [1/N]*N, args=(sigma), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(sigma),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


def rb(sigma, budget=None):
    '''
    风险预算模型：输入协方差矩阵和预算比例
    '''
    if budget:
        coef = np.divide(1, budget)
    else:
        coef = 1 / np.ones(len(sigma))

    def func(x, sigma, sign=1.0):
        N = len(sigma)
        R = sigma.dot(x) * (coef)
        S = sigma.dot(x).dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / S
        return sign * O

    N = len(sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    res = minimize(func, [1/N]*N, args=(sigma),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


def optim(func):
    '''
    通用优化函数：输入目标函数函数
    '''
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    res = minimize(func, [1/N]*N, args=(rho),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x
