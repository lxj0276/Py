# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:35:48 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file = 'C:/Users/s_zhangyw/Desktop/minute_000300.csv'

df = pd.read_csv(file, index_col=0, encoding="GBK", parse_dates=[0])
df = df.between_time('9:30', '15:00')

day_file = 'C:/Users/s_zhangyw/Desktop/day_000300.csv'

df_day = pd.read_csv(day_file, usecols=[2, 3], index_col=0, parse_dates=[0], encoding="GBK")
df_day = df_day['2015':]
df_day.columns=['Ret']

df_st = (df.pctchange.abs() * 100) / (df.volume ** 0.5)

df_q = pd.Series(name='q')
dateList = df_day.index.tolist()
for i in range(100, 828):
    date = dateList[i]
    date_ = dateList[i-20:i]
    data = df[date_[0]:date_[-1]]
    data = data[['close', 'volume', 'pctchange']]
    data['st'] = (data.pctchange.abs() * 100) / (data.volume ** 0.5)
    data = data.sort_values(by='st')
    data['v_cumsum'] = data.volume.cumsum()
    smart_i = len(data[data.v_cumsum < (0.5 * data.v_cumsum[-1])])
    Q = data.iloc[:smart_i, :].close.mean() / data.iloc[smart_i:, :].close.mean() 
    df_q[date] = Q

df_ret = df_day.rolling(10).apply(lambda x: (1 + x/100).prod() - 1).shift(5) * 100
df_ret = df_ret['2015-06-03':].Ret

df_q.corr(df_ret)
plt.scatter(df_q.rolling(600).apply(lambda x: x[-1] / x.mean()), df_ret)
df_ret[df_ret>0] =1
df_ret[df_ret<0] =0
df_ret.corr(df_q.rolling(600).apply(lambda x: x[-1] / x.mean()))
plt.scatter(df_q, df_ret)

y = '2018'
plt.scatter(df_q[y:], df_ret[y:])




#############################################
# -*- coding: utf-8 -*-
# 收盘价数据 
# 周期暂定k=20, h=5

file = 'C:/Users/s_zhangyw/Desktop/minute_000300.csv'

df = pd.read_csv(file, index_col=0, encoding="GBK", parse_dates=[0])
df = df.between_time('9:30', '15:00')

day_file = 'C:/Users/s_zhangyw/Desktop/day_000300.csv'
df_close = pd.read_csv(day_file, index_col=0, encoding="GBK", parse_dates=[0])
sr_ret = df_close.close.pct_change(5).shift(5)
dateList = sr_ret['20150601':'20180528'].index.date.tolist()


# 聪明钱因子Q
df['vwap'] = (df.open + df.close) / 2.0
df['smart'] = df.pctchange.abs() / df.volume ** 0.5
k = 10
sr_q = pd.Series(name='Q')
for i in range(k, len(dateList)):
    date = dateList[i]
    data = df[dateList[i-k]:dateList[i-1]]
    data = data.sort_values(by='smart')
    data['v_cumsum'] = data.volume.cumsum()
    smart_i = len(data[data.v_cumsum < (0.2 * data.v_cumsum[-1])])
    Q = data.iloc[:smart_i, :].vwap.mean() / data.iloc[smart_i:, :].vwap.mean() 
    sr_q[date] = Q

#  


# 基于日内模型的动量因子M
# 采集时点9:30, 10:30, 11:30, 13:00, 14:00, 15:00
def at_time(minute, sr):
    res = sr.between_time(minute, minute)
    res.index = res.index.date
    return res
k=20
sr_pct = df.pctchange
time = ['09:30', '10:30', '11:29', '13:01', '14:00', '15:00']
sr0930 = at_time(time[0], sr_pct)
sr1030 = at_time(time[1], sr_pct)
sr1130 = at_time(time[2], sr_pct)
sr1300 = at_time(time[3], sr_pct)
sr1400 = at_time(time[4], sr_pct)
sr1500 = at_time(time[5], sr_pct)

M0 = (sr0930 - sr1500.shift(1)).rolling(k).sum()
M1 = (sr1030 - sr0930).rolling(k).sum()
M2 = (sr1130 - sr1030).rolling(k).sum()
M3 = (sr1400 - sr1300).rolling(k).sum()
M4 = (sr1500 - sr1400).rolling(k).sum()

w = [0.04, 0.001, 1, -0.09, 0.14]
sr_m = w[0] * M0 + w[1] * M1 + w[2] * M2 + w[3] * M3 + w[4] * M4
sr_m = sr_m.fillna(0)
sr_m.name='M'

sr_q, sr_m, sr_ret = sr_q[-720:], sr_m[-720:], sr_ret[-720:] 

sr_q.corr(sr_ret)
sr_m.corr(sr_ret)

plt.scatter(sr_m, sr_ret)


from sklearn.metrics import classification_report 
from sklearn.metrics import precision_recall_curve, roc_curve, auc


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = pd.concat([sr_q, sr_m], axis=1)
model.fit(X, sr_ret)
sr_p = model.predict(X)
plt.scatter(sr_p, sr_ret)

from sklearn import svm
model = svm.SVR(kernel='poly')
X = pd.concat([sr_q, sr_m], axis=1)
model.fit(X, sr_ret)
sr_p = model.predict(X)
plt.scatter(sr_p, sr_ret)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X = pd.concat([sr_q, sr_m], axis=1)
y = np.where(sr_ret>0, 1, 0)
model.fit(X, y)
sr_p = model.predict(X)
