# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import feather as ft
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR, DynamicVAR

indexFile = 'C:/Users/s_zhangyw/Desktop/ETF.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], encoding='GBK')
df_rtn = df['涨跌幅(%)'].unstack().dropna()
df_exrtn = df_rtn.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]


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
df_predict = predict(df_exrtn)

df_predict5 = predict(df_exrtn, 4, 20)

df_predict20 = predict(df_exrtn20, 6, 12)

m=DynamicVAR(df_rtn, lag_order=1, window=None, window_type='expanding', trend='c', min_periods=None)
'''

###  Risk Parity Model
return_p = df_exrtn.rolling(12, min_periods=12).mean().shift(1)
return_p = return_p.applymap(lambda x: 100 * ((1 + x / 100.0) ** 50 -1))
vol_p = df_exrtn.ewm(alpha=0.8, min_periods=12).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (50 ** 0.5))
corr_p = df_exrtn.shift(1).expanding(12).corr()

# 逐月进入优化模型
l_date = vol_p.dropna().index.tolist()
l_sigma = []
for m in l_date:
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_sigma.append(sigma)

w = list(map(rp, l_sigma))    
df_w = pd.DataFrame(w, index=l_date, columns=df_exrtn.columns)
y = (df_rtn * df_w)
y = y.iloc[:, 1:]
y=y.dropna().sum(axis=1)
z=(y/100 + 1).cumprod()
z.plot()

x = df_rtn["中证全指"]
(x/100 + 1).cumprod().plot()


	ES = df_exrtn.rolling(10).apply(lambda x: -1 * np.sort(x)[:int(len(x) * 0.2)].mean())
	weight = ES.apply(lambda x: (1 / x) / (1 / x).sum(), axis=1)
	sr_rtn = (df_rtn * weight.shift(1)).mean(axis=1)
	sr_value = (1 + sr_rtn/100).cumprod()
	sr_value.plot()
    
########################################
    ####################################
    ####################################
########################################
df_rtn20 = df_close.iloc[::20, :].pct_change()
df_rtn20_cum = df_rtn20.rolling(4).apply(lambda x: np.prod(1 + x)).dropna() - 1
df_vol = df_rtn20.rolling(4).std()
sr_ew_rtn = df_rtn20.mean(axis=1).dropna()
df_corr = df_rtn20.rolling(4).corr(sr_ew_rtn)


######################################################
######################################################

df_rtn20 = df_close.iloc[::20, 1:].pct_change()
df_rtn20_cum = df_rtn20.rolling(4).apply(lambda x: np.prod(1 + x)).dropna()
df_post = df_rtn20_cum.apply(lambda x: np.where(
    x.rank(ascending=False) <= 4, 1/4, 0), axis=1)
sr_rtn = (df_post.shift(1) * df_rtn20).sum(axis=1).dropna()
sr_value = (1 + sr_rtn).cumprod()
sr_value.plot()
