# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
from scipy import stats

# 动量策略函数
def pos2value(df, df_pos, h):
    df_pos = df_pos.dropna().iloc[::h, :]
    df_tmp = df.copy()
    df_tmp.iloc[:, :] = 0
    df_pos = (df_pos + df_tmp).shift(1).fillna(method='ffill')
    sr_rtn = (df * df_pos).dropna().sum(axis=1)
    sr_value = (1 + sr_rtn).cumprod()

    return sr_value



# 1.1.1 绝对动量， 标准TSM


def taa_111(df, k, h, delta=60/61):
    df_rtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_vol = df_rtn.ewm(alpha=delta).std()
    df_pos = df_rtn.applymap(lambda x: 1 if x > 0 else 0)
    df_pos = df_pos / df_vol
    df_pos = df_pos.apply(lambda x: x / sum(x) if sum(x) else 0, axis=1)
   
    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 1.1.2 绝对动量， 相关性调整-CF-TSM+CVol


def taa_112(df, k, h, delta, sigma, l_rho):
    N = df.shape[1]
    df_rtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_rho = df_rtn.rolling(l_rho).corr()
    df_vol = df_rtn.ewm(alpha=delta).std()
    df_pos = df_rtn.applymap(lambda x: 1 if x > 0 else 0)
    sr_rho_avg = (df_rho.mean(axis=1).unstack().mean(axis=1) - 1 / N) / 2
    sr_cf = (N / (1 + (N - 1) * sr_rho_avg)) ** 0.5
    df_pos = (df_pos / df_vol).apply(lambda x: x /
                                     sum(x) if sum(x) else 0, axis=1)
    df_pos = df_pos.apply(lambda x: x / len(x[x > 0]) if sum(x) else 0, axis=1)
    df_pos = df_pos.apply(lambda x: x * sr_cf) * sigma

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 绝对动量， 方向概率RSM


def taa_113(df, k, h, delta, q):
    df_p = df.rolling(k).apply(lambda x: len(x[x > 0]) / k)
    df_vol = df_rtn.ewm(alpha=delta).std()

    df_pos = df_p.applymap(lambda x: 1 if x > q else 0)
    df_pos = df_pos / df_vol
    df_pos = df_pos.apply(lambda x: x / sum(x) if sum(x) else 0, axis=1)

    sr_value = pos2value(df, df_pos, h)
    return sr_value

# 相对动量， 标准XSM


def taa_121(df, k, h, n):
    df_rtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)

    df_pos = df_rtn.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)
    df_pos = df_pos.apply(lambda x: x / sum(x) if sum(x) else 0, axis=1)

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 　相对动量， CVX-XSM


def taa_122(df, k, h, n):
    N = df.shpae[1]
    X = [np.ones(N), np.range(N) + 1, (np.range(N) + 1) ** 2]
    df_rtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_gama = df.rolling(k).apply(lambda x: np.linalg.lstsq(X, y=x)[0][2])

    df_pos1 = df_rtn.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)
    df_pos2 = df_gama.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)
    df_pos = df_pos1 * df_pos2

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 相对动量， 广义GSM


def taa_123(df, k, h, n):
    w1, w2, w3 = 1.0, 0.5, 0.5
    df_rtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_vol = df_rtn.ewm(alpha=delta).std()
    df_rho = df_rtn.rolling(k).corr()
    df_rho_avg = df_rho.mean(axis=1).unstack()

    df_gsm = w1 * df_rtn.rank(ascending=False, axis=1) + w2 * df_vol.rank(
        ascending=True, axis=1) + w3 * df_rho_avg.rank(ascending=True, axis=1)
    df_pos = df_gsm.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 相对动量， 崭新FSM


def taa_124(k1, k2, h, n):
    df_rtn1 = df.rolling(k1).apply(lambda x: np.product(1 + x) - 1)
    df_rtn2 = df.rolling(k2).apply(lambda x: np.product(1 + x) - 1)

    df_pos1 = df_rtn1.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)
    df_pos2 = df_rtn2.apply(lambda x: 1 if x > x.sort(reverse=False)[int(len(x) / n)], axis=1)

    df_pos = df_pos1 * df_pos2

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 因子配置，趋势信号


def taa_211(df_close, n, l=[5, 20, 50, 100, 200]):
    df_rtn = df_close.pct_change(1)
    for i in l:

        pass

# TODO 212 213 214

# 因子配置， Max


def taa_215(df, k, h, n):
    df_max = df.rolling(k).max()
    df_pos = df_max.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# 因子配置-高阶矩


def taa_216(df, k, h, n):
    df_skew = df.rolling(k).skew()
    df_kurt = df.rolling(k).kurt()

    df_pos1 = df_skew.apply(lambda x: 1 if x > x.sort(reverse=False)[int(len(x) / n)], axis=1)
    df_pos2 = df_kurt.apply(lambda x: 1 if x > x.sort(reverse=True)[int(len(x) / n)], axis=1)

    df_pos = df_po1 * df_pos2

    sr_value = pos2value(df, df_pos, h)

    return sr_value

# TODO 217 218 需要指数数据
