# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
from comnfuncs import indics
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()
df_close20 = df_close.iloc[::20, 1:]
df_rtn20 = df_close20.pct_change(1)


def abm(sr_rtn, M=2):
    sr_rtn_sum = sr_rtn.rolling(M).sum()
    sr_pos = sr_rtn_sum.apply(lambda x: 1 if x > 0 else 0)
    sr_value = (1 + sr_pos.shift(1) * sr_rtn).cumprod().dropna()
    return indics(sr_value)[-1]


df_sp = pd.DataFrame()
for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24]:
    sr_tmp = df_rtn20.apply(lambda x: abm(x, m))
    sr_tmp.name = m
    df_sp = pd.concat([df_sp, sr_tmp], axis=1)
    
# 夏普比最优的参数基本在4， 5， 6
M = 4
df_rtn_sum = df_rtn20.rolling(M).sum().dropna()
df_pos = df_rtn_sum.applymap(lambda x: 1 if x > 0 else 0).fillna(0)
sr_value = (1 + (df_pos.shift(1) * df_rtn20).mean(axis=1)).cumprod().dropna()
print(indics(sr_value))
sr_value.plot()

# 不等权
M = 4
df_rtn_sum = df_rtn20.rolling(M).sum().dropna()
df_pos = df_rtn_sum[df_rtn_sum > 0].fillna(0)
df_pos = df_pos.apply(lambda x: x / sum(x), axis=1)
sr_value = (1 + (df_pos.shift(1) * df_rtn20).sum(axis=1)).cumprod().dropna()
print(indics(sr_value))
sr_value.plot()

# 超额收益
df_logrtn = np.log(df_close.iloc[::20, :]).diff(1)
df_exrtn = df_logrtn.dropna().apply(lambda x: x - x[0], axis=1).iloc[:, 1:]
M = 4
df_rtn_sum = df_exrtn.rolling(M).sum().dropna()
df_pos = df_rtn_sum[df_rtn_sum > 0].fillna(0)
df_pos = df_pos.apply(lambda x: x / sum(x), axis=1)
sr_value = (1 + (df_pos.shift(1) * df_rtn20).sum(axis=1)).cumprod().dropna()
print(indics(sr_value))
sr_value.plot()