# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:08:39 2018

@author: zhangyw49
"""
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.optimize import minimize

def optimize(func, func_deriv, *args):

    N = len(args[0])
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)})
    res = minimize(func, [1.0/N]*N, args=args, jac=func_deriv, constraints=cons, bounds=[(0, 1)] * N,
                   method='SLSQP', tol=1e-18, options={'ftol':1e-18,'disp': False, 'maxiter': 1000})

    return res.x


#######################################################################################

def max_sharpe(r, Sigma):
    ''' 最大夏普比

    Parameters
    ----------
    r : array of shape (n)
        收益向量
    Sigma : array of shape (n, n)
        协方差矩阵 

    Returns
    -------
    w* : array of shape (n)
        最优权重向量
    '''

    def func(x, r, Sigma, sign=-1.0):
        s = x.dot(Sigma).dot(x)
        res = x.dot(r) / s ** 0.5
        return res * sign

    def func_deriv(x, r, Sigma, sign=-1.0):
        s = x.dot(Sigma).dot(x)
        res = (r * s - x.dot(r) * Sigma.dot(x)) / s ** 1.5
        return res * sign

    return optimize(func, func_deriv, r, Sigma)


def min_vol(Sigma):
    ''' 最小波动率

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
        res = (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign
    
    def func_deriv(x, Sigma, sign=1.0):
        res = Sigma.dot(x) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign
    
    return optimize(func, func_deriv, Sigma)


def mean_variance(r, Sigma, lmd=2.5):
    ''' 均值方差优化

    Parameters
    ----------
    r : array of shape (n)
        收益向量

    Sigma : array of shape (n, n)
        协方差矩阵

    lmd : float
        风险厌恶系数 

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, r, Sigma, lmd, sign=-1.0):
        res = x.dot(r) - 0.5 * lmd * x.dot(Sigma).dot(x)
        return res * sign

    def func_deriv(x, r, Sigma, lmd, sign=-1.0):
        res = r - lmd * Sigma.dot(x)
        return res * sign
    
    return optimize(func, func_deriv, r, Sigma, lmd)


def target_variance(r, Sigma, tar_sigma=0.005):
    ''' 均值方差优化: 目标波动率

    Parameters
    ----------
    r : array of shape (n)
        收益向量

    Sigma : array of shape (n, n)
        协方差矩阵

    tar_sigma : float
        目标波动率 

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, r, Sigma, tar_sigma, sign=-1.0):
        return x.dot(r) * sign

    def func_deriv(x, r, Sigma, lmd, sign=-1.0):
        return r * sign

    N = len(Sigma)
    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)},
            {'type': 'ineq',
             'fun': lambda x: np.array(tar_sigma ** 2 - x.dot(Sigma).dot(x)),
             'jac': lambda x: np.array(-2 * Sigma.dot(x))})
    res = minimize(func, [1.0/N]*N, args=(r, Sigma, tar_sigma), jac=func_deriv, constraints=cons, bounds=[(0, 1)] * N,
                   method='SLSQP', tol=1e-18, options={'ftol':1e-18,'disp': False, 'maxiter': 1000})

    return res.x

def mv_resample(r, Sigma, k=2):
    ''' 均值方差优化

    Parameters
    ----------
    r : array of shape (n)
        收益向量

    Sigma : array of shape (n, n)
        协方差矩阵

    k : int
        重采样次数 

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''
    np.random.seed(123)
    wgt = np.zeros(len(r))
    for i in range(k):
        smp = np.random.multivariate_normal(r, Sigma, 10)
        r_, Sigma_ = smp.mean(axis=0), np.cov(smp.T)
        wgt += target_variance(r_, Sigma_)
    weight = wgt / k
    
    return weight


def black_litterman(r, Sigma, w_mkt, P, Q, Omega, lmd=2.5, tau=0.5):
    ''' Black_Litterman

    Parameters
    ----------
    r : array of shape (n)
        收益向量

    Sigma : array of shape (n, n)
        协方差矩阵

    P : array of shape (n)
        观点向量
        
    Q : array of shape (n, n)
        观点收益率矩阵
    
    Omega ：array of shape (n, n)
        观点误差矩阵
    
    tau : float
        观点比例因子    

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    Pai = lmd * (Sigma.dot(w_mkt))
    er_L = pinv(pinv(tau * Sigma) + P.dot(pinv(Omega)).dot(P))
    er_R = pinv(tau * Sigma).dot(Pai) + P.dot(pinv(Omega)).dot(Q)
    ER = er_L.dot(er_R)
    Nsigma = pinv(pinv(tau * Sigma) + P.dot(Omega).dot(P))
    weight = mean_variance(ER, Nsigma+Sigma, lmd)
    
    return weight

    

def vol_parity(Sigma):
    ''' 波动率平价    

    Parameters
    ----------
    Sigma : array of shape (n, n)
        协方差矩阵

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''
    sigma = Sigma.diagonal()
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

    return optimize(func, None, Sigma)


def risk_budget(budget, Sigma):
    ''' 风险预算

    Parameters
    ----------
    Sigma : array of shape (n, n)
        协方差矩阵

    budget : array of shape (n)
        风险预算向量

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, Sigma, sign=1.0):
        RC = (x / budget) * Sigma.dot(x) / (x.dot(Sigma).dot(x)) ** 0.5
        res = ((np.tile(RC, (len(RC), 1)).T - RC) ** 2).sum().sum()
        return res * sign

    return optimize(func, None, Sigma)


def most_diversified(Sigma):
    ''' 最大分散化   

    Parameters
    ----------
    Sigma : array of shape (n, n)
        协方差矩阵

    Returns
    -------
    w* : array of shape (n)
        最优权重向量   
    '''

    def func(x, Sigma, sign=-1.0):
        sigma = Sigma.diagonal()
        res = x.dot(sigma) / (x.dot(Sigma).dot(x)) ** 0.5
        return res * sign

    def func_deriv(x, Sigma, sign=-1.0):
        sigma = Sigma.diagonal()
        s = x.dot(Sigma).dot(x)
        res = (sigma * s ** 0.5 - x.dot(sigma) * Sigma.dot(x) / s ** 0.5) / s
        return res * sign

    return optimize(func, func_deriv, Sigma)


def most_decorr(Rho):
    ''' 最大去相关性   

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

    def func_deriv(x, Rho, sign=1.0):
        res = 2 * Rho.dot(x)
        return res * sign

    return optimize(func, func_deriv, Rho)


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

    return optimize(func, None, Sigma)


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
    df_rtn = pd.read_csv('rtn.csv', index_col=0, parse_dates=[0], encoding='GBK')
    df_rtn = df_rtn.fillna(0)
    # 收益率和协方差的预测
    rtn_p = df_rtn.shift(1).rolling(4).mean()
    rho_p = df_rtn.shift(1).rolling(4).corr()
    cov_p = df_rtn.shift(1).rolling(4).cov()

    # 转换为list
    l_day = rtn_p.dropna().index.tolist()
    l_r = []
    l_Rho = []
    l_Sigma = []
    for d in l_day:
        r = np.array(rtn_p.loc[d])
        Rho = np.array(rho_p.loc[d])
        Sigma = np.array(cov_p.loc[d])
        l_r.append(r)
        l_Rho.append(Rho)
        l_Sigma.append(Sigma)
    
    # 用最大夏普比优化
    l_weights = weights_solver("risk_parity", l_Sigma)

    # 回测净值
    df_pos = pd.DataFrame(l_weights, rtn_p.dropna().index,
                          columns=df_rtn.columns)

