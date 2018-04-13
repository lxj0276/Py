# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""
import numpy as np
import pandas as pd
import feather as ft
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR, DynamicVAR

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], encoding='GBK')
df_close = df['收盘价'].unstack()

df_rtn = df_close.pct_change(1).iloc[:, 1:].dropna()
df_logrtn = np.log(df_close).diff(1).dropna()
df_exrtn = df_logrtn.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]

df_close5 = df_close.iloc[::5, :]
df_rtn5 = df_close5.pct_change(1).iloc[:, 1:].dropna()
df_logrtn5 = np.log(df_close5).diff(1).dropna()
df_exrtn5 = df_logrtn5.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]

df_close20 = df_close.iloc[::20, :]
df_rtn20 = df_close20.pct_change(1).iloc[:, 1:].dropna()
df_logrtn20 = np.log(df_close20).diff(1).dropna()
df_exrtn20 = df_logrtn20.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]


ft.write_dataframe(df_exrtn5.dropna(), 'C:/Users/s_zhangyw/Desktop/exrtn.ft')

def var(arr, lag):
	model = VAR(arr)
	results = model.fit(maxlags=lag, ic='aic')
	lag_order = results.k_ar
	fct = results.forecast(arr[-lag_order:], 1)
	return fct

def predict(df, lag=20, roll=250):
	idx = df.index[roll:]
	clmn = df.columns
	arr = df.values
	brr = np.zeros((arr.shape[0]-roll, arr.shape[1]))
	for i in range(roll, len(arr)):
		data = arr[i-roll:i, :]
		brr[i-roll, :] = var(data, lag)
	return pd.DataFrame(brr, columns=clmn, index=idx)

def netvalue(df_predict, df_rtn):
    df_rank = df_predict.rank(axis=1, ascending=False)
    df_buy = df_rank[df_rank<=1].shift(1)
    df_totalrtn = (df_buy * df_rtn).mean(axis=1)

    df_value = (df_totalrtn + 1 - 0.001).cumprod()
    df_value.plot()
    

'''    
df_predcit = predict(df_exrtn)

df_predict5 = predict(df_exrtn5, 12, 24)

df_predict20 = predict(df_exrtn20, 6, 12)

m=DynamicVAR(df_rtn, lag_order=1, window=None, window_type='expanding', trend='c', min_periods=None)
'''

###  Risk Parity Model

df_rtn = df_exrtn5.dropna()
ES = df_rtn.rolling(100).apply(lambda x: -1 * np.sort(x)[:int(len(x) * 0.2)].mean())
weight = ES.apply(lambda x: (1 / x) / (1 / x).sum(), axis=1)
sr_rtn = (df_rtn5 * weight.shift(1)).mean(axis=1)
sr_value = (1 + sr_rtn).cumprod()
sr_value.plot()




vol_p = df_exrtn5.ewm(alpha=0.2, min_periods=12).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (50 ** 0.5))

corr_p = df_exrtn5.shift(1).expanding(12).corr()

# 估计协方差矩阵
l_date = vol_p.dropna().index.tolist()
l_sigma = []
for m in l_date:
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_sigma.append(sigma)
    
w = list(map(rp_s, l_sigma))    
df_w = pd.DataFrame(w, index=l_date, columns=df_exrtn5.columns)
y = (df_rtn5 * df_w)
y = y.iloc[:, 1:]
y=y.dropna().sum(axis=1)
z=(y + 1).cumprod()
z.plot()

z_rp=z/z[0]
fig(z_rp, w, 'RP Model')
