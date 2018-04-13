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

# 利用跟踪误差做风险平价

indexFile = 'C:/Users/s_zhangyw/Desktop/index.csv'
df = pd.read_csv(indexFile, index_col=[2, 1], parse_dates=[2], encoding='GBK')
df_close = df['收盘价'].unstack()

# 计算跟踪误差
N = 5  # 换仓周期
M = 50  # lookback长度
df_cls = df_close.iloc[::N, :]
df_rtn = df_cls.pct_change(1).iloc[:, 1:].dropna()
df_logrtn = np.log(df_cls).diff(1).dropna()
df_exrtn = df_logrtn.apply(lambda x: x - x[0], axis=1).iloc[:, 1:]


def rp_simple(df_risk):
    df_pos = df_risk.apply(lambda x: (1 / x) / sum(1 / x), axis=1).dropna()
    sr_rtn = (df_rtn * df_pos).sum(axis=1).dropna()
    sr_value = (1 + sr_rtn).cumprod()
    return sr_value


# 方法一：波动率
df_vol = df_exrtn.rolling(50).std()
sr_value = rp_simple(df_vol)
plt.plot(sr_value)

# 方法二：VaR
q = 0.2
df_var = df_exrtn.rolling(50).apply(lambda x: -1 * np.sort(x)[int(len(x) * q)])
sr_value = rp_simple(df_var)
plt.plot(sr_value)

# 方法三：ES
q = 0.2
df_es = df_exrtn.rolling(50).apply(
    lambda x: -1 * np.sort(x)[:int(len(x) * 0.2)].mean())
sr_value = rp_simple(df_es)
plt.plot(sr_value)

# 方法四：半方差
df_vol = df_exrtn.rolling(50).apply(lambda x: np.sqrt((x[x < 0] ** 2).mean()))
sr_value = rp_simple(df_vol)
plt.plot(sr_value)


plt.legend(['Vol', 'VaR', 'ES', 'Semi'])
