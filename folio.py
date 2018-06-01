# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:23:15 2018

@author: s_zhangyw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from betas import *

df_rtn = pd.read_csv('rtn.csv', index_col=0, parse_dates=[0], encoding='GBK')
df_rtn = df_rtn.fillna(0)
# 收益率和协方差的预测
rtn_p = df_rtn.shift(1).rolling(24).mean()
rho_p = df_rtn.shift(1).rolling(24).corr()
cov_p = df_rtn.shift(1).rolling(24).cov()

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

# 等权重模型
pos_ew = df_rtn.copy()
pos_ew.iloc[:, :] = 1 / pos_ew.shape[1]
value_ew = (df_rtn.mean(axis=1) + 1).cumprod()
value_ew.plot()
# 其他模型
def solver(func_name, *args):
    l_weights = weights_solver(func_name, *args)
    df_pos = pd.DataFrame(l_weights, rtn_p.dropna().index,
                      columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, 4)
    sr_value.plot()
    return df_pos, sr_value

## 最小波动率
pos_mv, value_mv = solver('min_vol', l_Sigma)

## 波动率平价
pos_emv, value_emv = solver('vol_parity', l_Sigma)

## 风险平价
pos_rp, value_rp = solver('risk_parity', l_Sigma)

## 风险预算
pos_rb, value_rb = solver('risk_budget', l_r, l_Sigma)

## 最大分散化
pos_md, value_md = solver('most_diversified', l_Sigma)

## 最大去相关性
pos_decorr, value_decorr = solver('most_decorr', l_Rho)

## 最大夏普比
pos_msr, value_msr = solver('max_sharpe', l_r, l_Sigma)

## 均值-方差优化
pos_mvo, value_mvo = solver('mean_variance', l_r, l_Sigma)

## ReSample
pos_smp, value_smp = solver('mv_resample', l_r, l_Sigma)

## Black-Litterman: 动量观点

# 读取收益
df_rtn = pd.read_csv('rtn.csv', index_col=0, parse_dates=[0], encoding='GBK')
df_rtn = df_rtn.dropna()
# 收益率和协方差的预测
rtn_p = df_rtn.shift(1).rolling(24).mean()
rtn_sum = df_rtn.shift(1).rolling(24).sum()
cov_p = df_rtn.shift(1).rolling(24).cov()
# 转换为list
l_day = rtn_p.dropna().index.tolist()
l_w_mkt = [np.ones(df_rtn.shape[1]) / df_rtn.shape[1]] * len(l_day)
l_P = [np.ones(df_rtn.shape[1])] * len(l_day)
l_Omega =  [np.diag(np.ones(df_rtn.shape[1]))] * len(l_day)

l_r = []
l_Q = []
l_Sigma = []
for d in l_day:
    r = np.array(rtn_p.loc[d])
    q = np.array(rtn_sum.loc[d])
    Sigma = np.array(cov_p.loc[d])
    l_r.append(r)
    l_Q.append(q)
    l_Sigma.append(Sigma)

pos_bl, value_bl = solver('black_litterman', l_r[60:], l_Sigma[60:], l_w_mkt[60:], l_P[60:], l_Q[60:], l_Omega[60:])


def black_litterman(r, Sigma, w_mkt, P, Q, Omega, lmd=2.5, tau=0.5):

    Pai = lmd * (Sigma.dot(w_mkt))
    er_L = inv(inv(tau * Sigma) + P.dot(inv(Omega)).dot(P))
    er_R = inv(tau * Sigma).dot(Pai) + P.dot(inv(Omega)).dot(Q)
    ER = er_L.dot(er_R)
    Nsigma = inv(inv(tau * Sigma) + P.dot(Omega).dot(P))
    weight = mean_variance(ER, Nsigma+Sigma, lmd)
    print(r)
    return weight
