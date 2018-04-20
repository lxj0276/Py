# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

# 光大证券：风险平价 + 动量价值 + RSRS = 资产配置
# 大类资产： 大、中、小盘， 港股，转债， 利率债5Y、10Y， 信用债5Y， 现金
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from rp import rp
from comnfuncs import indics


#　样本内优化（IS： 2005-2010， OS： 2011-2018）
csvFile = 'C:/Users/s_zhangyw/Desktop/assets.csv'
df = pd.read_csv(csvFile, index_col=0, parse_dates=[0], encoding='GBK').iloc[::, :-1]
df = df[:'2011']
df_rtn = df.pct_change(1).dropna()
df_lrtn = np.log(df).diff(1).dropna()


# 持仓周期 5天
h = 5
df_h = df.iloc[::h, :]
df_rtn_h = df_h.pct_change(1).dropna()
df_lrtn_h = np.log(df_h).diff(1).dropna()

# 风险平价
## 均值，波动率的预测

### 收益率最优参数
for i in range(1, 25):
    return_p = df_rtn_h.rolling(i).mean().shift(1)
    score = return_p.corrwith(df_rtn_h).mean()
    print(i, score)
### i = 3
return_p = df_rtn_h.rolling(3).mean().shift(1)
return_p = return_p.applymap(lambda x: ((1 + x) ** 12 -1))
### 波动率最优参数
for i in range(1, 25):
    vol_p = df_rtn_h.ewm(span=i).std().shift(1)
    vol_r = df_rtn.rolling(h).std().iloc[4::5, :]
    score = vol_p.corrwith(vol_r).mean()
    print(i, score)
### i = 15
vol_p = df_rtn_h.ewm(span=15).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (12 ** 0.5))
corr_p = df_rtn_h.shift(1).expanding(15).corr()

## 均值向量，协方差矩阵
l_month = return_p.dropna().index.tolist()
l_u = []
l_sigma = []
for m in l_month:
    u = np.array(return_p.loc[m])
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_u.append(u)
    l_sigma.append(sigma)

## 风险平价模型
w = list(map(rp, l_sigma))    
df_w = pd.DataFrame(w, index=l_month, columns=df.columns)
y = (df_rtn_h * df_w)
y=y.dropna().sum(axis=1)
z=(1 + y).cumprod()
z_rp=z/z[0]
z_rp.plot()
indics(z_rp, h)

## 动量因子

## 动量效应寻优
def tsm_best(sr_lrtn, L, H):
    # L:look bakck H:holding periods
    #L = 24
    #H = 12
    arr_sp = np.zeros((L, H))
    for l in range(1, L+1):
        tsm = sr_lrtn.rolling(l).sum().apply(lambda x: 1 if x > 0 else 0).shift(1)
        for h in range(1, H+1):            
            pos = tsm.dropna().iloc[::h]
            df = pd.concat([sr_lrtn, pos], axis=1).fillna(method="ffill").dropna()
            sr_lr = df.iloc[:, 0] * df.iloc[:, 1]
            sr_value = sr_lr.cumsum().apply(lambda x: np.exp(x))
            sp = indics(sr_value, f=h)[2]
            arr_sp[l-1, h-1] = sp
    return arr_sp
##
i = 5
sr = df_lrtn_h.iloc[:, i]
sp = tsm_best(sr, 48, 1)
# 最优H全部为1， L暂拟为6
L = 3
k = 3
df_tsm = df_lrtn_h.rolling(L).sum().applymap(lambda x: k if x > 0 else 1 / k).shift(1)
df_w_m = (df_w * df_tsm).apply(lambda x: x / sum(x), axis=1)
y = (df_rtn_h * df_w_m)
y=y.dropna().sum(axis=1)
z=(1 + y).cumprod()
z=z/z[0]
z.plot()
indics(z, h)

# K寻优
for k in range(1, 10):
    df_tsm = df_lrtn_h.rolling(L).sum().applymap(lambda x: k if x > 0 else 1 / k).shift(1)
    df_w_m = (df_w * df_tsm).apply(lambda x: x / sum(x), axis=1)
    y = (df_rtn_h * df_w_m)
    y=y.dropna().sum(axis=1)
    z=(1 + y).cumprod()
    print(indics(z, h)[2])
## k=3


## TODO 价值因子

## TODO RSRS 指标
    
# 全样本

csvFile = 'C:/Users/s_zhangyw/Desktop/assets.csv'
df = pd.read_csv(csvFile, index_col=0, parse_dates=[0], encoding='GBK').iloc[::, :-1]
df_rtn = df.pct_change(1).dropna()
df_lrtn = np.log(df).diff(1).dropna()


# 持仓周期 5天
h = 5
df_h = df.iloc[::h, :]
df_rtn_h = df_h.pct_change(1).dropna()
df_lrtn_h = np.log(df_h).diff(1).dropna()

# 风险平价
## 均值，波动率的预测

return_p = df_rtn_h.rolling(3).mean().shift(1)
return_p = return_p.applymap(lambda x: ((1 + x) ** 12 -1))

vol_p = df_rtn_h.ewm(span=15).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (12 ** 0.5))
corr_p = df_rtn_h.shift(1).expanding(15).corr()

## 均值向量，协方差矩阵
l_month = return_p.dropna().index.tolist()
l_u = []
l_sigma = []
for m in l_month:
    u = np.array(return_p.loc[m])
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_u.append(u)
    l_sigma.append(sigma)

## 风险平价模型
w = list(map(rp, l_sigma))    
df_w = pd.DataFrame(w, index=l_month, columns=df.columns)
y = (df_rtn_h * df_w)
y=y.dropna().sum(axis=1)
z=(1 + y).cumprod()
z_rp=z/z[0]
z_rp.plot()
indics(z_rp, h)

## 动量因子
L = 3
k = 3
df_tsm = df_lrtn_h.rolling(L).sum().applymap(lambda x: k if x > 0 else 1 / k).shift(1)
df_w_m = (df_w * df_tsm).apply(lambda x: x / sum(x), axis=1)
y = (df_rtn_h * df_w_m)
y=y.dropna().sum(axis=1)
z=(1 + y).cumprod()
z=z/z[0]
z.plot()
indics(z, h)
