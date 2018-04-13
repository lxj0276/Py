# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()
df_rtn = df_close.pct_change(1).iloc[:, 1:].dropna()
df_logrtn = np.log(df_close).diff(1)
df_exrtn = df_logrtn.dropna().apply(lambda x: x - x[0], axis=1).iloc[:, 1:]

# 等权重作基准EW
df_rtn20 = df_close.iloc[::20, 1:].pct_change()
sr_rtn = df_rtn20.mean(axis=1).dropna()
sr_ew_value = (1 + sr_rtn).cumprod()
plt.plot(sr_ew_value)

# 设定动量参数
N = 4  # 买入资产数量
M = 2  # lookback 2个月

# 相对动量R, N=4, lookback=4M
df_rtn20 = df_close.iloc[::20, 1:].pct_change()
df_rtn20_cum = df_rtn20.rolling(M).apply(lambda x: np.prod(1 + x)).dropna()
df_pos = df_rtn20_cum.apply(lambda x: np.where(
    (x.rank(ascending=False) <= N), 1/N, 0), axis=1)
sr_rtn = (df_pos.shift(1) * df_rtn20).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
plt.plot(sr_value)

# 加入绝对动量A, rmin=0
df_rtn20 = df_close.iloc[::20, 1:].pct_change()
df_rtn20_cum = df_rtn20.rolling(M).apply(lambda x: np.prod(1 + x)).dropna()
df_pos = df_rtn20_cum.apply(lambda x: np.where(
    (x.rank(ascending=False) <= N), 1/N, 0), axis=1)
df_pos[df_rtn20_cum < 1] = 0
sr_rtn = (df_pos.shift(1) * df_rtn20).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
plt.plot(sr_value)

# 再加入波动率V
df_rtn = df_close.iloc[:, 1:].pct_change()
df_rtn20 = df_close.iloc[::20, 1:].pct_change()
df_rtn20_cum = df_rtn20.rolling(M).apply(lambda x: np.prod(1 + x)).dropna()
df_vol = df_rtn.rolling(M * 20).std().iloc[::20, :].dropna()
df_Li = 1 * df_rtn20_cum.apply(lambda x: x.rank(ascending=False),
                               axis=1) + 0.5 * df_vol.apply(lambda x: x.rank(), axis=1)
df_pos = df_Li.apply(lambda x: np.where(x.rank() <= N, 1/N, 0), axis=1)
df_pos[df_rtn20_cum < 1] = 0
sr_rtn = (df_pos.shift(1) * df_rtn20).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
plt.plot(sr_value)

# 再加入相关系数C
df_rtn = df_close.iloc[:, 1:].pct_change()
df_rtn20 = df_close.iloc[::20, 1:].pct_change()
df_rtn20_cum = df_rtn20.rolling(M).apply(lambda x: np.prod(1 + x)).dropna()
df_vol = df_rtn.rolling(M * 20).std().iloc[::20, :].dropna()
df_corr = df_rtn.rolling(M * 20).corr()
df_corr_avg = df_corr.mean(axis=1).unstack().iloc[::20, :].dropna()
df_Li = 1 * df_rtn20_cum.apply(lambda x: x.rank(ascending=False), axis=1) + 0.5 * df_vol.apply(
    lambda x: x.rank(), axis=1) + 0.5 * df_corr_avg.apply(lambda x: x.rank(), axis=1)
df_pos = df_Li.apply(lambda x: np.where(x.rank() <= N, 1/N, 0), axis=1)
df_pos[df_rtn20_cum < 1] = 0
sr_rtn = (df_pos.shift(1) * df_rtn20).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
plt.plot(sr_value)

# 添加图例
plt.legend(['EW', 'R', 'A', 'V', 'C'])