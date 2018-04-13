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
from statsmodels.tsa.api import VAR

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()


def rtn(N, df):
    df_close = df.iloc[::N, :]
    df_rtn = df_close.pct_change(1).iloc[:, 1:].dropna()
    df_logrtn = np.log(df_close).diff(1).dropna()
    df_exrtn = df_logrtn.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]
    return df_rtn, df_exrtn


def var(arr, lag):
    model = VAR(arr)
    results = model.fit(maxlags=lag, ic='aic')
    lag_order = results.k_ar
    if lag_order:
        fct = results.forecast(arr[-lag_order:], 1)
    else:
        fct = arr[-1]
    return fct


def predict(df, lag, roll):
    idx = df.index[roll:]
    clmn = df.columns
    arr = df.values
    brr = np.zeros((arr.shape[0]-roll, arr.shape[1]))
    for i in range(roll, len(arr)):
        data = arr[i-roll:i, :]
        brr[i-roll, :] = var(data, lag)
    return pd.DataFrame(brr, columns=clmn, index=idx)


N = 5
A = 1
lag = 2
roll = 48
df_rtn, df_exrtn = rtn(N, df_close)
df_predict = predict(df_exrtn, lag, roll)
df_rank = df_predict.rank(axis=1, ascending=False)
df_pos = df_rank.apply(lambda x: np.where(x <= A, 1, 0))
sr_rtn = (df_pos.shift(1) * df_rtn).mean(axis=1)
sr_value = (sr_rtn + 1).cumprod()
plt.plot(sr_value)


''''
v1 = netvalue(1, lag=20, roll=250)
v2 = netvalue(5, lag=2, roll=48)
v3 = netvalue(20, lag=2, roll=48)
'''
