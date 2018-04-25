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

indexFile = 'C:/Users/s_zhangyw/Desktop/000300.csv'
df = pd.read_csv(indexFile, index_col=2, parse_dates=[2], encoding='GBK')
df_300 = df[['开盘价', '收盘价', '最高价', '最低价']]

N = 18
M = 600
df_cov = df[['最高价', '最低价']].rolling(N).cov().iloc[:, 1].unstack()
sr_RS = df_cov.iloc[:, 0] / df_cov.iloc[:, 1]
sr_Z = sr_RS.rolling(M).apply(lambda x: (x[-1] - x.mean()) / x.std())
sr_R = (df_cov.iloc[:, 0].rolling(N).corr(df_cov.iloc[:, 1])) ** 2
sr_RSRS = sr_Z * sr_R

df_cov = df[['收盘价', '开盘价']].rolling(N).cov().iloc[:, 1].unstack()
sr_RS_1 = df_cov.iloc[:, 0] / df_cov.iloc[:, 1]

sr_rtn = df['涨跌幅(%)'] / 100.0
sr_tsm = sr_rtn.rolling(N).apply(lambda x: (x + 1).prod() - 1)
H = 20
sr_rtn_sum = (sr_rtn.rolling(H).apply(lambda x: (x + 1).prod() - 1)).shift(H)
x=sr_RSRS.corr(sr_tsm)

plt.scatter(sr_RSRS, sr_rtn_sum)
df_RS = pd.concat([pd.cut(sr_RSRS,10),  sr_rtn_sum], axis=1)
df_RS.columns=['RS','Rtn']
df_RS.groupby(by='RS').mean()
df_RS.plot()
plt.twinx()

sr_RS_1[sr_RS_1>=0.8] = 1
sr_RSRS[sr_RSRS<=-1] = -1
sr_RSRS[sr_RSRS.abs()<1] = np.nan

plt.figure(figsize=(120,10))
plt.plot(df['收盘价'])
plt.twinx()
plt.plot(sr_RSRS, 'r')


plt.twinx()

