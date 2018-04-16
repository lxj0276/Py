# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()
df_rtn = df_close.pct_change(1).iloc[:, 1:].dropna()
df_lrtn = np.log(df_close.iloc[:, 1:]).diff(1).dropna()
df_lrtn_sum = df_lrtn.rolling(5).sum().shift(-4)

for i in range(20, 200):
    X = df_lrtn.iloc[i-20:i, :].T.values
    y = df_lrtn_sum.iloc[i, :].values
    reg = RandomForestRegressor(oob_score=True, n_jobs=4)
    reg.fit(X, y)
    print(reg.oob_score_)