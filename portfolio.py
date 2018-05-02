# -*- encoding: utf-8 -*-
'''
投资组合函数：均值方差，风险平价，风险预算
'''

import numpy as np
from scipy.optimize import minimize

def mv(u, Sigma, lmd=2.5):
    '''
    均值方差模型：输入均值向量，协方差矩阵和风险厌恶系数
    '''

    def func(x, u, Sigma, lmd, sign=-1.0):
        N = len(Sigma)
        L = 0
        for i in range(N):
            L = L + x[i] * u[i]
        R = 0
        for j in range(N):
            for k in range(N):
                R = R + x[j] * x[k] * Sigma[j][k]

        return sign * (L - lmd / 2.0 * R)

    def func_deriv(x, u, Sigma, lmd, sign=-1.0):
        N = len(Sigma)
        dfdx = []
        for i in range(N):
            R = 0
            for j in range(N):
                R = R + Sigma[i, j] * x[j]

            dfdxi = sign * (u[i] - lmd * R)
            dfdx.append(dfdxi)
        return np.array(dfdx)

    N = len(Sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    # res = minimize(func, [1/N]*N, args=(u, Sigma, lmd), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(u, Sigma, lmd),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})
    return res.x


def rp(Sigma):
    '''
    风险平价模型：输入协方差矩阵
    '''

    def func(x, Sigma, sign=1.0):
        N = len(Sigma)
        R = Sigma.dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / R.dot(x)
        return sign * O

    def func_deriv(x, Sigma, sign=1.0):
        N = len(Sigma)
        R = Sigma.dot(x)
        U = R.dot(x)

        def drv(i, k, r=R, s=Sigma, w=x):
            if i != k:
                return s[i][k] * w[i]
            else:
                return r.sum() + r[i]

        def drv_Sigma(k, s=Sigma, w=x):
            N = len(Sigma)
            D = 0
            for i in range(N):
                D = D + s[k, j] * w[j]
            return D

        def drv_total(i, k):
            return (drv(i, k) * U ** 0.5 - (drv_Sigma(k) * x[i] * R[i]) / (2 * U ** 0.5)) / U

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

    N = len(Sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    # res = minimize(func, [1/N]*N, args=(Sigma), jac=func_deriv, \
    res = minimize(func, [1/N]*N, args=(Sigma),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


def rb(Sigma, budget=None):
    '''
    风险预算模型：输入协方差矩阵和预算比例
    '''
    if budget:
        coef = np.divide(1, budget)
    else:
        coef = 1 / np.ones(len(Sigma))

    def func(x, Sigma, sign=1.0):
        N = len(Sigma)
        R = Sigma.dot(x) * (coef)
        S = Sigma.dot(x).dot(x)
        O = 0
        for i in range(N):
            for j in range(N):
                O = O + ((x[i] * R[i] - x[j] * R[j])) ** 2 / S
        return sign * O

    N = len(Sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})

    res = minimize(func, [1/N]*N, args=(Sigma),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


from numpy.linalg import inv
def bl(l_u, l_sigma, w_mkt=[0.05, 0.05, 0.1, 0.6, 0.1, 0.1]):
    lmd = 2.5
    tau = 0.5    
    l_w = []

    for i in range(len(l_u)):
        if i >= 3:
            P = np.array([0, 0, 0, 0, 0, 0])
            mnt = sum(l_u[i-2:i+1]) / 3.0
            u = mnt.argmax()
            d = mnt.argmin()
            P[u] = 1
            P[d] = -1
            Q = mnt[u] - mnt[d]
            Omega = np.eye(6) * Q
            
            
            u = l_u[i]
            sigma = l_sigma[i]
            pai = lmd * (sigma.dot(w_mkt))
            er_l = inv(inv(tau * sigma) + P.dot(inv(Omega)).dot(P))
            er_r = inv(tau * sigma).dot(pai) + P.dot(inv(Omega)).dot(Q)
            ER = er_l.dot(er_r)
            Nsigma = inv(inv(tau * sigma) + P.dot(Omega).dot(P))
        
            l_w.append(mv(ER, Nsigma+sigma, lmd))
        else:
            u = l_u[i]
            sigma = l_sigma[i]
            l_w.append(mv(u, sigma, lmd))
    return l_w

def bl_mv(l_u, l_sigma, l_w_mkt):
    lmd = 2.5
    tau = 0.5
    l_w = []

    
    
    for i in range(len(l_u)):
        if i >= 3:
            w_mkt = l_w_mkt[i]
            P = np.array([0, 0, 0, 0, 0, 0])
            mnt = sum(l_u[i-2:i+1]) / 3.0
            u = mnt.argmax()
            d = mnt.argmin()
            P[u] = 1
            P[d] = -1
            Q = mnt[u] - mnt[d]
            Omega = np.eye(6) * Q
            
            
            u = l_u[i]
            sigma = l_sigma[i]
            pai = lmd * (sigma.dot(w_mkt))
            er_l = inv(inv(tau * sigma) + P.dot(inv(Omega)).dot(P))
            er_r = inv(tau * sigma).dot(pai) + P.dot(inv(Omega)).dot(Q)
            ER = er_l.dot(er_r)
            Nsigma = inv(inv(tau * sigma) + P.dot(Omega).dot(P))
        
            l_w.append(mv(ER, Nsigma+sigma, lmd))
        else:
            u = l_u[i]
            sigma = l_sigma[i]
            l_w.append(mv(u, sigma, lmd))
    return l_w


# we need lmd tau Omega P Q
# w_mkt equal weight
# P Q Omega
# -*- coding: utf-8 -*-
from numpy.linalg import inv
def bl_rp(l_u, l_sigma, budget = [[1, 1, 1, 1, 1, 1]]):
    lmd = 2.5
    tau = 0.8   
    l_w = []
    l_w_mkt =  list(map(rb, l_sigma, budget * len(l_sigma)))
    
    for i in range(len(l_u)):
        if i >= 3:
            w_mkt = l_w_mkt[i]
            P = np.array([0, 0, 0, 0, 0, 0])
            mnt = sum(l_u[i-2:i+1]) / 3.0
            u = mnt.argmax()
            d = mnt.argmin()
            P[u] = 1
            P[d] = -1
            Q = mnt[u] - mnt[d]
            Omega = np.eye(6) * Q
            
            
            
            u = l_u[i]
            sigma = l_sigma[i]
            pai = lmd * (sigma.dot(w_mkt))
            er_l = inv(inv(tau * sigma) + P.dot(inv(Omega)).dot(P))
            er_r = inv(tau * sigma).dot(pai) + P.dot(inv(Omega)).dot(Q)
            ER = er_l.dot(er_r)
            Nsigma = inv(inv(tau * sigma) + P.dot(Omega).dot(P))
        
            l_w.append(mv(ER, Nsigma+sigma, lmd))
        else:
            u = l_u[i]
            sigma = l_sigma[i]
            l_w.append(mv(u, sigma, lmd))
    return l_w


# we need lmd tau Omega P Q
# w_mkt equal weight
# P Q Omega

