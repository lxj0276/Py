# -*- coding: utf-8 -*-
"""
投资组合优化函数
如果资产数目为n，则

Parameters
----------
r : array of shape (n)
    收益向量
sigma : array of shape (n)
    方差向量
Sigma : array of shape (n, n)
    协方差矩阵
Rho: array of shape (n, n)
    相关系数矩阵   

Returns
-------
w* : array of shape (n)
    最优权重向量  
"""


import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize(func, *args):

    N = len(args[0])
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})
    res = minimize(func, [1/N]*N, args=args, constraints=cons, bounds=[(0, 1)] * N,
                   method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x


#######################################################################################

def max_sharp(r, Sigma):
    ''' 最大夏普比

    Parameters
    ----------
    r : array of shape (n)
        收益向量
    Sigma : array of shape (n)
        协方差矩阵 

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, r, Sigma, sign=-1.0):
        res = x.dot(r) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, r, Sigma)


def min_vol(Sigma):
    ''' 最小波动率

    Parameters
    ----------
    Sigma : array of shape (n)
        协方差矩阵


    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, Sigma, sign=1.0):
        res = (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, Sigma)


def vol_parity(sigma):
    ''' 波动率平价    

    Parameters
    ----------
    sigma : array of shape (n)
        方差向量

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    return (1.0 / sigma) / (1.0 / sigma).sum()


def risk_parity(Sigma):
    ''' 风险平价    

    Parameters
    ----------
    Sigma : array of shape (n, n)
        协方差矩阵

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, Sigma, sign=1.0):
        RC = x * Sigma.dot(x) / (x.dot(Sigma).dot(x)) ** 0.5
        res = ((np.tile(RC, (len(RC), 1)).T - RC) ** 2).sum().sum()
        return res * sign

    return optimize(func, Sigma)


def most_diversified(sigma, Sigma):
    ''' 最大分散化   

    Parameters
    ----------
    sigma : array of shape (n)
        方差向量
    Sigma : array of shape (n, n)
        协方差矩阵

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, sigma, Sigma, sign=-1.0):
        res = x.dot(sigma) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    return optimize(func, sigma, Sigma)


def most_decorr(Rho):
    ''' 最大去相关系数    

    Parameters
    ----------
    Rho: array of shape (n, n)
        相关系数矩阵   

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''
    def func(x, Rho, sign=1.0):
        res = x.dot(Rho).dot(x)
        return res * sign

    return optimize(func, Rho)


def max_entropy(Sigma, k=2):
    ''' 最大熵   
    Parameters
    ----------
    Sigma : array of shape (n, n)
        协方差矩阵
    k : int
        约束条件个数
    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def safe_log(arr):
        brr = np.zeros_like(arr)
        brr = np.where(arr > 0, np.log(arr), 0)
        return brr

    def func(x, Sigma, sign=1.0):
        Sigma = (Sigma + Sigma.T) / 2.0
        eig_values, eig_vectors = np.linalg.eig(Sigma)

        descend_order = eig_values.argsort()[::-1]
        a = eig_values[descend_order]
        b = eig_vectors[:, descend_order]

        y = b.T.dot(x)
        v = (y**2) * a
        p = v / v.sum()
        res = (p * safe_log(p))[k:].sum()
        return res * sign

    return optimize(func, Sigma)


#######################################################################################

def weights_solver(method, *args):
    ''' 组合优化求解

    Parameters
    ----------
    method : str
        组合优化函数名称，例如："risk_parity"
    *args : list
        以list形式传入优化函数需要的参数序列

    Returns
    -------
    l_weights : list
        每期的最优权重 

    Notes
    -------
    传入*args时，需要与优化函数的位置参数顺序保持一致

    '''

    l_weights = list(map(eval(method), *args))
    return l_weights


#######################################################################################

# Demo
if __name__ == "__main__":

    # 读收益率数据
    df_rtn = pd.read_csv('rtn.csv', index_col=0)

    # 收益率和协方差的预测
    rtn_p = df_rtn.rolling(20).mean().shift(1)
    cov_p = df_rtn.rolling(20).cov().shift(1)

    # 转换为list
    l_month = rtn_p.dropna().index.tolist()
    l_r = []
    l_Sigma = []
    for m in l_month:
        u = np.array(rtn_p.loc[m])
        Sigma = np.array(cov_p.loc[m])
        l_r.append(u)
        l_Sigma.append(Sigma)

    # 用最大夏普比优化
    l_weights = weights_solver("max_sharp", l_r, l_Sigma)