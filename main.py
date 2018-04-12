#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:16:34 2018

@author: tober
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import optimize

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']




path_close = 'C:/Users/tober/Desktop/Close.xlsx'
df_close = pd.read_excel(path_close, skiprows=[0, 1, 3], index_col=0)
# df_close = df_close.iloc[:, [0,2,3,4,5,12,13,15]]
df_close = df_close.iloc[:, [0,3,4,5,12,15]]
df_close = df_close['2005':]
(df_close / df_close.iloc[0, :]).plot(figsize=(15,8), fontsize=20)
'''
plt.figure(figsize=(10,6), dpi=300)
plt.plot((df_close / df_close.iloc[0, :]))
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('各类资产历史表现')
plt.legend(df_close.columns)
'''
# 指标评价
'''
df = df_close / df_close.iloc[0, :]
df_indics = df.apply(indics)

'''

path_return = 'C:/Users/tober/Desktop/Return.xlsx'
df_return = pd.read_excel(path_return, skiprows=[0, 1, 3], index_col=0)
df_return.plot()
Cor = df_return.corr()
# 由于相关性太强，去掉部分资产
df_return = pd.read_excel(path_return, skiprows=[0, 1, 3], index_col=0)
df_return = df_return.iloc[:, [0,3,4,5,12,15]]
df_return = df_return.dropna()
#df_return.plot(figsize=(15,8))
Cor = df_return.corr()
'''
for i in range(6):
    plt.figure(dpi=100)
    df = df_return.iloc[:, i].sort_values()
    plt.hist(df, bins=30)
    plt.xlabel('收益率')
    plt.ylabel('频数')
    plt.title(df.name)
    plt.savefig('C:/Users/tober/Desktop/%s.png'%i)
'''

# 预测收益率、相关系数和波动率
return_p = df_return.rolling(6, min_periods=6).mean().shift(1)
return_p = return_p.applymap(lambda x: 100 * ((1 + x / 100.0) ** 12 -1))

vol_p = df_return.ewm(alpha=0.2, min_periods=7).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (12 ** 0.5))

corr_p = df_return.shift(1).expanding(7).corr()

# 逐月进入优化模型
l_month = vol_p.dropna().index.tolist()
l_u = []
l_sigma = []
for m in l_month:
    u = np.array(return_p.loc[m])
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_u.append(u)
    l_sigma.append(sigma)


# 等权重模型
'''
w = [[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]] * len(l_month)

#w = [[1/20, 1/20, 2/20, 12/20, 2/20, 2/20]] * len(l_month) 
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_ew=z/z[0]
fig(z_ew, w, 'EW Model')
indics(z_ew)

'''

# 均值方差模型
'''
w = list(map(mv, l_u, l_sigma))
w = list(map(clear, w))    
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_mv=z/z[0]
fig(z_mv, w, 'MV Model')
indics(z_mv)

z=pd.concat([z_ew, z_mv], axis=1)
plt.figure(dpi=300)
plt.plot(z)
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('EW Model & MV Model')
plt.legend(['EW Model', 'MV Model'])
plt.savefig('7.png')

'''
# 固定重模型
'''
w = [[1/20, 1/20, 2/20, 12/20, 2/20, 2/20]] * len(l_month) 
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_cw=z/z[0]
fig(z_cw, w, 'CW Model')
indics(z_cw)

'''

# BL模型

'''
w=bl(l_u, l_sigma)
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_bl=z/z[0]
fig(z_bl, w, 'BL Model')
indics(z_bl)

z=pd.concat([z_ew, z_bl], axis=1)
plt.figure(dpi=300)
plt.plot(z)
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('CW Model & BL Model')
plt.legend(['CW Model', 'BL Model'])
plt.savefig('7.png')
'''

# 风险平价模型
'''
w = list(map(rp, l_sigma))    
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_rp=z/z[0]
fig(z_rp, w, 'RP Model')

indics(z_rp)
'''
# 风险预算模型
'''
budget = [[1, 1, 1, 0.1, 1, 1]]
w = list(map(rb, l_sigma, budget * len(l_sigma)))   
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z_rb=(y/100 + 1).cumprod()
fig(z_rb, w, 'RB Model')
indics(z_rb)

z=pd.concat([z_rp, z_rb], axis=1)
plt.figure(dpi=300)
plt.plot(z)
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('RP Model & RB Model')
plt.legend(['RP Model', 'RB Model'])
plt.savefig('7.png')
'''

# 积极风险均衡
'''
w=bl_rp(l_u, l_sigma)
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z_rl=z/z[0]
fig(z_rl, w, 'RP-BL Model')
indics(z_rl)

z=pd.concat([z_rp, z_rl], axis=1)
plt.figure(dpi=300)
plt.plot(z)
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('RP Model & RP-BL Model')
plt.legend(['RP Model', 'RP-BL Model'])
plt.savefig('7.png')
'''

# END
'''
z=pd.concat([z_ew, z_mv, z_bl, z_rp, z_rl], axis=1)
plt.figure(dpi=300)
plt.plot(z)
plt.xlabel('年份')
plt.ylabel('净值')
plt.title('资产配置模型对比')
plt.legend(['EW Model', 'MV Model', 'BL Model', 'RP Model', 'RP-BL Model'])
plt.savefig('7.png')
'''













# 积极风险均衡 + 预算
'''
budget = [[1, 1, 1, 0.1, 1, 1]]
w=bl_rp(l_u, l_sigma, budget)
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z=z/z[0]
fig(z, w, 'RB-BL Model')
indics(z)
'''


#l_w_mkt = w
# BL-MV
'''
w=bl_mv(l_u, l_sigma, l_w_mkt)
df_w = pd.DataFrame(w, index=l_month, columns=df_return.columns)
y = (df_return * df_w)
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z=z/z[0]
fig(z, w, 'BL-MV Model')
indics(z)
'''