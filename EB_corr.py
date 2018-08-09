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



# =============================================================================
# from WindPy import w
# 
# w.start()
# 
# def getWsd(wsd):
#     df = pd.DataFrame(wsd.Data).T
#     df.index = wsd.Times
#     df.columns = wsd.Codes 
#     return df
# 
# 
# wsd = w.wsd("881001.WI,000016.SH,CBA00102.CS,CBA00202.CS,CBA00662.CS", "pct_chg", "2000-01-01", "2018-07-29", "Period=M")
# # 全A， 50， 新综合， 综合， 10年国债 
# df_idx = getWsd(wsd)
# =============================================================================



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


# 按日计算相关系数

# =============================================================================
# from WindPy import w
# w.start()
# 
# def getWsd(wsd):
#     df = pd.DataFrame(wsd.Data).T
#     df.index = wsd.Times
#     df.columns = wsd.Codes 
#     return df
# 
# wsd = w.wsd("881001.WI, CBA00102.CS", "pct_chg", "2000-01-01", "2018-07-31", "Period=D")
# df_idx = getWsd(wsd)
# w.stop()
# 
# =============================================================================
df = pd.read_csv('df.csv', index_col=0, parse_dates=[0])
sr_corr = df.resample('M').apply(lambda x: x.corr().iloc[0, 1])
sr_corr.name = "corr"

sr_beta = df.resample('M').apply(lambda x: x.corr().iloc[0, 1] / x.iloc[:, 0].std()
          * x.iloc[:, 1].std())


sr_corr_y = df.resample('Y').apply(lambda x: x.corr().iloc[0, 1])
sr_beta_y =  df.resample('Y').apply(lambda x: x.corr().iloc[0, 1] / x.iloc[:, 0].std()
          * x.iloc[:, 1].std())


# 处理利率和通胀、增长和货币数据
df_rate = pd.read_csv('rate.csv', index_col=0, parse_dates=[0])
df_rate = df_rate.resample('M').mean()
df_cpi = pd.read_csv('cpi.csv', index_col=0, parse_dates=[0])
df_m2 = pd.read_csv('m2.csv', index_col=0, parse_dates=[0])
df_gdp = pd.read_csv('gdp.csv', index_col=0, parse_dates=[0])

# 对齐
sr_corr = sr_corr['2002':'2018-06']
sr_rate = df_rate.Rate['2002':'2018-06']
sr_cpi = df_cpi.CPI['2002':'2018-06']
sr_m2 = df_m2.M2['2002':'2018-06']
sr_gdp = df_gdp.GDP['2002':'2018-06']

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



# =============================================================================
# 全部重来
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn import linear_model

#from WindPy import w
#w.start()
# 
#def getWsd(wsd):
#     df = pd.DataFrame(wsd.Data).T
#     df.index = wsd.Times
#     df.columns = wsd.Codes 
#     return df
# 
#wsd_d = w.wsd("881001.WI, CBA00102.CS", "pct_chg", "2000-01-01", "2018-07-31", "Period=D")
#wsd_m = w.wsd("881001.WI, CBA00102.CS", "pct_chg", "2000-01-01", "2018-06-30", "Period=M")
#
#df_d = getWsd(wsd_d)
#df_m = getWsd(wsd_m)
#w.stop()

#df_d.dropna().to_csv('df_d.csv')
#df_m.dropna().to_csv('df_m.csv')


df_m = pd.read_csv('df_m.csv', index_col=0, parse_dates=[0])

lag = 12
corr = df_m['881001.WI'].rolling(lag).corr(df_m['CBA00102.CS'])
beta = corr / df_m['881001.WI'].rolling(lag).std() * df_m['CBA00102.CS'].rolling(lag).std()
corr.name = 'corr'
beta.name = 'beta'

y_e = df_m['881001.WI']
y_b = df_m['CBA00102.CS']

# 一图胜千言

# fig1
# =============================================================================
# sort = df_m[df_m['881001.WI']<0].sort_values(by='881001.WI').values
# 
# fig, ax = plt.subplots(dpi=200)
# ax.bar(np.arange(len(sort)), sort[:,0], label='Stock')
# ax.set_ylabel("Stock (%)")
# ax.legend(loc=3)
# ax2 = ax.twinx()
# ax2.plot(np.arange(len(sort)), sort[:,1], 'r', label='Bond')
# ax2.axhline(0, color='k')
# ax2.set(ylim=[-2, 2])
# ax2.set_ylabel("Bond (%)")
# ax2.legend(loc=4)
# =============================================================================

# fig2
# =============================================================================
# sort = df_m.sort_values(by='881001.WI').values
# for i in range(5):
#     s = i * len(sort) // 5
#     e = (i + 1) * len(sort) // 5
#     plt.hist(sort[s:e, 1], bins=7)
#     plt.show()
# =============================================================================


# 扩展因子； 增长，通胀，利率， 流动性
# GDP增速，CPI， DR007， M2， 

# 处理利率和通胀、增长和货币数据
df_rate = pd.read_csv('rate.csv', index_col=0, parse_dates=[0])
df_rate = df_rate.resample('M').mean()
df_cpi = pd.read_csv('cpi.csv', index_col=0, parse_dates=[0])
df_m2 = pd.read_csv('m2.csv', index_col=0, parse_dates=[0])
df_gdp = pd.read_csv('gdp.csv', index_col=0, parse_dates=[0])

# 对齐
corr = corr['2003':'2018-06']
rate = df_rate.Rate['2003':'2018-06']
cpi = df_cpi.CPI['2003':'2018-06']
m2 = df_m2.M2['2003':'2018-06']
gdp = df_gdp.GDP['2003':'2018-06']

df_tmp = pd.concat([m2, gdp], axis=1)
df_tmp = df_tmp.fillna(method='bfill')
gdp = df_tmp.GDP

y_e = y_e['2003':'2018-06']
y_b = y_b['2003':'2018-06']


# 很强的自相关性，回归不如自回归
reg = linear_model.LinearRegression()
X = np.vstack([rate.values, cpi.values, m2.values, gdp.values])
y = corr.values
reg.fit(X.T, y)
y_p = reg.predict(X.T)
plt.plot(corr.index, y, corr.index, y_p)
plt.plot(corr.index, y, corr.index[1:], y[:-1])

# 转入R
data = np.vstack([corr.values, rate.values, cpi.values, m2.values, 
                  gdp.values, y_e.values, y_b.values])
col_names = ['corr', 'rate', 'cpi', 'm2', 'gdp', 'y_e', 'y_b']
Rdata = pd.DataFrame(data.T, index=corr.index, columns=col_names)
Rdata.index.name = 'Date'
Rdata.to_csv('Rdata.csv')

# 
reg = linear_model.LinearRegression()
X = np.vstack([rate.diff().values, cpi.diff().values, m2.diff().values, gdp.diff().values])
y = corr.values
reg.fit(X[:, 1:].T, y[1:])
y_p = reg.predict(X[:, 1:].T)
plt.plot(corr.index[1:], y[1:], corr.index[1:], y_p)
