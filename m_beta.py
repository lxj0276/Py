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

# 利用扣除贝塔因素的超额收益做动量组合

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()

# 计算跟踪误差
N = 5  # 换仓周期，天
M = 200  # lookback长度，天
A = 4  # 资产数目
df_cls = df_close.iloc[::N, :]
df_rtn = df_cls.pct_change(1).iloc[:, 1:].dropna()

df_logrtn = np.log(df_close).diff(1)
sr_idxrtn = df_logrtn.iloc[:, 0]
df_beta = df_logrtn.rolling(M).cov(sr_idxrtn) / df_logrtn.rolling(M).var()
df_exrtn = (df_logrtn - df_beta.apply(lambda x: x * sr_idxrtn)
            ).iloc[::N, 1:].dropna()
df_pos = df_exrtn.apply(lambda x: np.where(
    x.rank(ascending=False) <= A, 1/A, 0), axis=1)
df_pos[df_exrtn.rolling(M).sum() < 0] = 0
sr_rtn = (df_pos.shift(1) * df_rtn).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
plt.plot(sr_value)
