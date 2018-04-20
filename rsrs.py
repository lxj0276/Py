# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

# 光大证券：风险平价 + 动量价值 + RSRS = 资产配置
# 大类资产： 大、中、小盘， 港股，转债， 利率债5Y、10Y， 信用债5Y， 现金
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from rp import rp
from comnfuncs import indics

csvFile = 'C:/Users/s_zhangyw/Desktop/assets.csv'
df = pd.read_csv(csvFile, index_col=0, parse_dates=[
                 0], encoding='GBK').iloc[::, :-1]
df_rtn = df.pct_change(1).dropna()
df_lrtn = np.log(df).diff(1).dropna()


# 持仓周期 5天
h = 5
df_h = df.iloc[::h, :]
df_rtn_h = df_h.pct_change(1).dropna()
df_lrtn_h = np.log(df_h).diff(1).dropna()

# 风险平价
# 均值，波动率的预测
return_p = df_rtn_h.rolling(12).mean().shift(1)
return_p = return_p.applymap(lambda x: ((1 + x) ** 12 - 1))
vol_p = df_rtn_h.ewm(span=12, min_periods=12).std().shift(1)
vol_p = vol_p.applymap(lambda x: x * (12 ** 0.5))
corr_p = df_rtn_h.shift(1).expanding(12).corr()

l_month = vol_p.dropna().index.tolist()

# 均值向量，协方差矩阵
l_u = []
l_sigma = []
for m in l_month:
    u = np.array(return_p.loc[m])
    v = np.diag(vol_p.loc[m])
    rho = np.array(corr_p.loc[m])
    sigma = v.dot(rho).dot(v)
    l_u.append(u)
    l_sigma.append(sigma)

# 风险平价模型
w = list(map(rp, l_sigma))
df_w = pd.DataFrame(w, index=l_month, columns=df.columns)
y = (df_rtn_h * df_w)
y = y.dropna().sum(axis=1)
z = (1 + y).cumprod()
z_rp = z/z[0]
z_rp.plot()
indics(z_rp, h)

# 动量因子
l = 3
k = 10
df_tsm = df_lrtn_h.rolling(l).sum().applymap(
    lambda x: k if x > 0 else 1 / k).shift(1)
df_w_m = (df_w * df_tsm).apply(lambda x: x / sum(x), axis=1)
y = (df_rtn_h * df_w_m)
y = y.dropna().sum(axis=1)
z = (1 + y).cumprod()
z_mom = z/z[0]
z_mom.plot()
indics(z_mom, h)

# 价值因子


# RSRS指标
rsfile = "C:/Users/s_zhangyw/Desktop/rs_300.csv"
df_rs = pd.read_csv(rsfile, index_col=0, parse_dates=[0], encoding="GBK")
