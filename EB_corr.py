# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


'''
from WindPy import w

w.start()

def getWsd(wsd):
    df = pd.DataFrame(wsd.Data).T
    df.index = wsd.Times
    df.columns = wsd.Codes 
    return df


wsd = w.wsd("881001.WI,000016.SH,CBA00102.CS,CBA00202.CS,CBA00662.CS", "pct_chg", "2000-01-01", "2018-07-29", "Period=M")
# 全A， 50， 新综合， 综合， 10年国债 
df_idx = getWsd(wsd)
'''


df_idx = pd.read_csv('idx.csv', index_col=0, parse_dates=[0])
df_idx = df_idx['2002':].iloc[:, [0, 2]]

lag = 12
sr_corr = df_idx['881001.WI'].rolling(lag).corr(df_idx['CBA00102.CS'])
sr_beta = sr_corr / df_idx['881001.WI'].rolling(lag).std() * df_idx['CBA00102.CS'].rolling(lag).std()

sr_corr.dropna(inplace=True)
sr_beta.dropna(inplace=True)
# 分年度相关性
sr_corr_gp = sr_corr.groupby(lambda x: x.year).agg(lambda x: x[-1]).dropna()
sr_beta_gp = sr_beta.groupby(lambda x: x.year).agg(lambda x: x[-1]).dropna()

#881001指数表现
sr_881 = (1 + df_idx['881001.WI'] / 100).cumprod()

plt.plot(sr_corr, 'r')
plt.twinx()
plt.plot(sr_881, 'b')

plt.plot(sr_beta, 'g')
plt.twinx()
plt.plot(sr_881, 'b')

plt.figure(figsize=(25,8))
plt.plot(sr_corr, 'g')
plt.twinx()
plt.plot(sr_881, 'b')

def corr_spearman(df, lag):
    sr = pd.Series(index=df.index, name='corr')
    for i in range(lag, len(df)):
        sr.iloc[i] = df.iloc[i-lag:i].corr(method='spearman').iloc[0, -1]
    return sr

sr_corr_sm = corr_spearman(df_idx, lag)
    
plt.plot(sr_corr.index, sr_corr, sr_corr_sm)
plt.grid()


# 处理利率和通胀数据
df_rate = pd.read_csv('rate.csv', index_col=0, parse_dates=[0])
df_rate = df_rate.resample('M').mean()
df_cpi = pd.read_csv('cpi.csv', index_col=0, parse_dates=[0])

# 对齐
sr_corr.name='Corr'
sr_corr = sr_corr['2003':'2018-06']
sr_rate = df_rate.Rate['2003':'2018-06']
sr_cpi = df_cpi.CPI['2003':'2018-06']

# 回归
from sklearn import linear_model
reg = linear_model.LinearRegression()
X = np.vstack([sr_rate.values, sr_cpi.values])
y = sr_corr.values
reg.fit(X.T, y)
y_p = reg.predict(X.T)
plt.plot(sr_corr.index, y, sr_corr.index, y_p)

# 处理增长和货币数据
df_m2 = pd.read_csv('m2.csv', index_col=0, parse_dates=[0])
df_gdp = pd.read_csv('gdp.csv', index_col=0, parse_dates=[0])
sr_m2 = df_m2.M2['2003':'2018-06']
sr_gdp = df_gdp.GDP['2003':'2018-06']
df_tmp = pd.concat([sr_m2, sr_gdp], axis=1)
df_tmp = df_tmp.fillna(method='bfill')
sr_gdp = df_tmp.GDP

#
reg = linear_model.LinearRegression()
X = np.vstack([sr_rate.values, sr_cpi.values, sr_m2.values, sr_gdp.values])
y = sr_corr.values
reg.fit(X.T, y)
y_p = reg.predict(X.T)
plt.plot(sr_corr.index, y, sr_corr.index, y_p)


# 转入R处理协整
df_r = pd.DataFrame(np.vstack([X, y]).T, index=sr_corr.index, columns=['rate', 
                    'cpi', 'm2', 'gdp', 'corr'])

df_r.to_csv('df_r.csv')

#
reg = linear_model.LinearRegression()
X = np.vstack([sr_cpi.values, sr_m2.values])
y = sr_corr.values
reg.fit(X.T, y)
y_p = reg.predict(X.T)
plt.plot(sr_corr.index, y, sr_corr.index, y_p)