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
df = df.between_time('9:30', '12:00')

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

df_ret = df_day.rolling(10).appl y(lambda x: (1 + x/100).prod() - 1).shift(5) * 100
df_ret = df_ret['2015-06-03':].Ret

df_q.corr(df_ret)
plt.scatter(df_q.rolling(600).apply(lambda x: x[-1] / x.mean()), df_ret)
df_ret[df_ret>0] =1
df_ret[df_ret<0] =0
df_ret.corr(df_q.rolling(600).apply(lambda x: x[-1] / x.mean()))
plt.scatter(df_q, df_ret)

y = '2018'
plt.scatter(df_q[y:], df_ret[y:])
