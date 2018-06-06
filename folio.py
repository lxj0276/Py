# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:23:15 2018

@author: s_zhangyw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from betas import *
from Evaluation import evaluation

df_rtn = pd.read_csv('rtn.csv', index_col=0, parse_dates=[0], encoding='GBK')
df_rtn = df_rtn.fillna(0)
# 收益率和协方差的预测
rtn_p = df_rtn.shift(1).rolling(102).mean()
rho_p = df_rtn.shift(1).rolling(102).corr()
cov_p = df_rtn.shift(1).rolling(102).cov()

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

# 仓位、净值
def solver(func_name, *args):
    l_weights = weights_solver(func_name, *args)
    df_wgt = pd.DataFrame(l_weights, rtn_p.dropna().index,
                      columns=df_rtn.columns)
    df_pos, sr_value = pos2value(df_rtn, df_wgt, 1)
    sr_value.plot()
    return df_pos, sr_value
# 风险贡献
def risk_contribution(df_pos):
    df_rc = df_pos.copy()
    df_Sigma = df_rtn.rolling(102).cov()
    for idx, w in df_pos.iterrows():
        v = w.values
        Sigma = df_Sigma.loc[idx]
        rc = v * (v.dot(Sigma)) / (v.dot(Sigma).dot(v)) #** 0.5
        df_rc.loc[idx] = rc
    return df_rc


# 等权重模型
pos_ew = df_rtn['2009':].copy()
pos_ew.iloc[:, :] = 1 / pos_ew.shape[1]
value_ew = (df_rtn['2009':].mean(axis=1) + 1).cumprod()
value_ew.plot()


# 其他模型
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

'''
## Black-Litterman: 动量观点

# 读取收益
df_rtn = pd.read_csv('rtn.csv', index_col=0, parse_dates=[0], encoding='GBK')
df_rtn = df_rtn.dropna()
# 收益率和协方差的预测
rtn_p = df_rtn.shift(1).rolling(4).mean()
rtn_sum = df_rtn.shift(1).rolling(4).sum()
cov_p = df_rtn.shift(1).rolling(4).cov()
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

pos_bl, value_bl = solver('black_litterman', l_r, l_Sigma, l_w_mkt, l_P, l_Q, l_Omega)
'''

## 图
value_list = [value_ew, value_mv, value_emv, value_rp, value_rb, value_md, value_decorr, value_msr, value_mvo, value_smp]
pos_list = [pos_ew, pos_mv, pos_emv, pos_rp, pos_rb, pos_md, pos_decorr, pos_msr, pos_mvo, pos_smp]

## 净值
plt.figure(dpi=300)
for value in value_list:
    plt.plot(value)
plt.legend(['EW', 'MV', 'EMV', 'RP', 'RB', 'MD', 'DeCorr', 'MSR', 'MVO', 'ReSmp'])
## 评价指标
for pos in pos_list:
    E = evaluation(pos.dropna(), df_rtn['2009':])
    print(E.YearRet())
for pos in pos_list:
    E = evaluation(pos.dropna(), df_rtn['2009':])
    print('####\n', E.CAGR(), E.Ret_Roll_1Y()[0], E.Ret_Roll_3Y()[0], E.Stdev(), E.Skew(), E.MaxDD(), E.MaxDD_Dur(), 
          E.VaR(), E.SR(), E.Calmar(), E.RoVaR(), E.Hit_Rate(), E.Gain2Pain(), sep='\n')

#面积图、箱线图
for pos in pos_list:
    pos.plot.area()
    pos.plot.box(rot=60)
    

#风险贡献
for pos in pos_list:
    i = 0 
    
    pos = pos_list[i]
    df_rc = risk_contribution(pos.dropna())
    plt.stackplot(df_rc.index, df_rc.values.T)
    plt.legend(df_rc.columns)
    df_rc.plot.box(rot=60)
    i = i + 1
 

# 止损类策略
from rule import *
# 原始
df_value = (df_rtn + 1).cumprod()
df_value.plot()
#
df_value = (df_pos * df_rtn + 1).cumprod()
df_value.plot()
# 止损
df_pos = df_rtn.apply(lambda x: stop_loss(x, -0.05, 0.01))

# 目标波动率
df_sigma = df_rtn.rolling(102).std() * (50) ** 0.5     #年化波动率
df_pos = df_sigma.apply(lambda x: target_vol(x, 0.10, 1.0, 0.10))

# 回撤控制
df_CVaR = df_rtn.rolling(102).apply(lambda x: np.sort(x)[:int(102 * 0.05)].mean() * (50) ** 0.5)
df_pos = df_CVaR.apply(lambda x: dd_control(x, -0.05, 1.0))

## CPPI
df_pos = df_rtn.apply(lambda x: CPPI(x, 10, 0.5))

## TIPP
df_pos = df_rtn.apply(lambda x: TIPP(x, 10, 0.5))

## OBPI
pass

## VaR套补
pass

## margrabe资产交换
pass