# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

'''
TS-MOM
by MOP

时间序列动量，传统意义上的动量仅仅指横截面的相对动量
lookback  k M
holding   h M
如果超额收益为正，持有
如果为负， 持有现金
仓位与波动率成反比
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comnfuncs import indics
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()

codestr = '''
df_rtn = (df_close.iloc[::20, 1:]).pct_change(1).dropna()
df_logrtn = np.log(df_close.iloc[::20, 1:]).diff(1).dropna()
df_vol = np.log(df_close.iloc[::, 1:]).diff(1).ewm(span=60).std() * (250 ** 0.5)
df_rtn_sum = df_logrtn.rolling(k).sum().dropna()

df_pos = df_rtn_sum.applymap(lambda x: 1 if x > 0 else 0)
df_pos = df_pos * df_vol.loc[df_pos.index, :]
df_pos = df_pos.apply(lambda x: (1 / x) / sum(1 / x), axis=1)

sr_rtn = (df_pos.shift(1) * df_rtn).sum(axis=1)
sr_value = (1 + sr_rtn).cumprod()
sr_value.plot()
print(indics(sr_value))
'''

k = 5; h = 1; exec(codestr)

# 风险调整的时间序列动量

codestr = '''
df_rtn = (df_close.iloc[::20, 1:]).pct_change(1).dropna()
df_logrtn = np.log(df_close.iloc[::, 1:]).diff(1).dropna()
df_vol = np.log(df_close.iloc[::, 1:]).diff(1).ewm(span=60).std() * (250 ** 0.5)
df_lrtn_vol = df_logrtn / df_vol
df_rtn_sum = df_lrtn_vol.rolling(k*20).sum().iloc[::20, :].dropna()

df_pos = df_rtn_sum.applymap(lambda x: 1 if x > 0 else 0)
df_pos = df_pos * df_vol.loc[df_pos.index, :]
df_pos = df_pos.apply(lambda x: (1 / x) / sum(1 / x), axis=1)

sr_rtn = (df_pos.shift(1) * df_rtn).sum(axis=1)
sr_value = (1 + sr_rtn).cumprod()
sr_value.plot()
print(indics(sr_value))
'''

k = 5; h = 1; exec(codestr)