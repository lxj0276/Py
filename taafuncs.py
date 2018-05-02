# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:46:45 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import portfolio


# 策略1.1.1—绝对动量—标准TSM

def pos2value(df_rtn, df_pos, h):
    assert (df_pos >= 0).all().all(), "negtive weight"
    df_pos = df_pos.dropna().iloc[::h, :]
    df_pos = df_pos.apply(lambda x: x / sum(x) if sum(x)
                          else x, raw=False, axis=1)
    df_tmp = df_rtn.copy()
    df_tmp.iloc[:, :] = 0
    df_pos = (df_pos + df_tmp).shift(1).fillna(method='ffill')
    sr_rtn = (df_rtn * df_pos).dropna().sum(axis=1)
    sr_value = (1 + sr_rtn).cumprod()
    sr_value.plot()

    return sr_value


# 1.1.1 绝对动量， 标准TSM


def taa_111(df_rtn, k, h, delta=60/61):
    df_srtn = df_rtn.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_vol = df_rtn.ewm(alpha=delta).std()
    df_pos = df_srtn.applymap(lambda x: 1 if x > 0 else 0)
    df_pos = df_pos / df_vol

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略1.1.2—绝对动量—相关性调整CF-TSM+Cvol

def taa_112(df_rtn, k, h, delta, sigma, l_rho):
    N = df.shape[1]
    df_srtn = df_rtn.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_rho = df_rtn.rolling(l_rho).corr()
    df_vol = df.ewm(alpha=delta).std()
    df_pos = df_srtn.applymap(lambda x: 1 if x > 0 else 0)
    sr_rho_avg = (df_rho.mean(axis=1).unstack().mean(axis=1) - 1 / N) / 2
    sr_cf = (N / (1 + (N - 1) * sr_rho_avg)) ** 0.5
    df_pos = (df_pos / df_vol).apply(lambda x: x /
                                     sum(x) if sum(x) else 0, axis=1)
    df_pos = df_pos.apply(lambda x: x * sr_cf) * sigma

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略1.1.3—绝对动量—方向概率RSM

def taa_113(df_rtn, k, h, delta, q):
    df_p = df_rtn.rolling(k).apply(lambda x: len(x[x > 0]) / k)
    df_vol = df.ewm(alpha=delta).std()

    df_pos = df_p.applymap(lambda x: 1 if x > q else 0)
    df_pos = df_pos / df_vol

    sr_value = pos2value(df_rtn, df_pos, h)
    return sr_value



# 策略1.2.1—相对动量—标准XSM

def taa_121(df_rtn, k, h, n):
    N = df.shape[1]
    df_srtn = df_rtn.rolling(k).apply(lambda x: np.product(1 + x) - 1)

    df_pos = df_srtn.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                          int(N / n)] else 0 for i in x], axis=1)

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略1.2.2—相对动量—凸型Cvx-XSM

def taa_122(df_rtn, k, h, n):
    N = df_rtn.shape[1]
    X = np.array([np.ones(k), np.arange(k) + 1, (np.arange(k) + 1) ** 2])
    df_srtn = df.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_gama = df.rolling(k).apply(
        lambda y: np.linalg.lstsq(X.T, y, rcond=-1)[0][2])

    df_pos1 = df_srtn.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                           int(N / n)] else 0 for i in x], axis=1)
    df_pos2 = df_gama.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                            int(N / n)] else 0 for i in x], axis=1)
    df_pos = df_pos1 * df_pos2

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略1.2.3—相对动量—广义GSM

def taa_123(df_rtn, k, h, delta, n):
    w1, w2, w3 = 1.0, 0.5, 0.5
    N = df_rtn.shape[1]
    df_srtn = df_rtn.rolling(k).apply(lambda x: np.product(1 + x) - 1)
    df_vol = df.ewm(alpha=delta).std()
    df_rho = df_rtn.rolling(k).corr()
    df_rho_avg = df_rho.mean(axis=1).unstack()

    df_gsm = w1 * df_srtn.rank(ascending=False, axis=1) + w2 * df_vol.rank(
        ascending=True, axis=1) + w3 * df_rho_avg.rank(ascending=True, axis=1)
    df_pos = df_gsm.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                          int(N / n)] else 0 for i in x], axis=1)

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略1.2.4—相对动量—崭新FSM

def taa_124(df_rtn, k1, k2, h, n):
    N = df.shape[1]
    df_rtn1 = df_rtn.rolling(k1).apply(lambda x: np.product(1 + x) - 1)
    df_rtn2 = df_rtn.rolling(k2).apply(lambda x: np.product(1 + x) - 1)

    df_pos1 = df_rtn1.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                            int(N / n)] else 0 for i in x], axis=1)
    df_pos2 = df_rtn2.apply(lambda x: [1 if i > sorted(x, reverse=False)[
                            int(N / n)] else 0 for i in x], axis=1)

    df_pos = df_pos1 * df_pos2

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.1.1—因子配置—趋势信号

def taa_211(df_close, n, h, L=[5, 20, 50, 100, 200]):
    df_rtn = df_close.pct_change(1)
    arr_pa = np.zeros((len(L), df_close.shape[0], df_close.shape[1]))
    for i in range(len(L)):
        df_pal = df_close.rolling(L[i]).mean() / df_close
        arr_pa[i] = df_pal.values

    df_frtn = df_rtn.rolling(h).apply(lambda x: (1 + x).prod() - 1).shift(-h)
    arr_frtn = df_frtn.values

    arr_beta = np.zeros((arr_frtn.shape[0], len(L) + 1))

    for i in range(len(arr_frtn)):
        y = arr_frtn[i]
        X = np.vstack([arr_pa[:, i, :], np.ones(len(y))])
        beta = np.linalg.lstsq(X.T, y, rcond=-1)[0]
        arr_beta[i] = beta

        pass



# 策略2.1.2—因子配置—流动性波动

def taa_212(df_rtn, df_amt, k, h, n):
    df_rtn_sum = df_rtn.rolling(k).apply(lambda x: (1 + x).prod() - 1)
    df_amt_sum = df_amt.rolling(k).sum()

    df_DPIOF = df_rtn_sum.abs() / df_amt_sum

    pass



# 策略2.1.3—因子配置—低beta

def taa_213(df_rtn, sr_rtn_m, k, h, n):
    N = df_rtn.shape[1]
    df_std = df_rtn.rolling(250 * 1).std()
    sr_std_m = sr_rtn_m.rolling(250 * 1).std()
    df_corr = df_rtn.rolling(250 * 5).apply(lambda x: x.corr(sr_rtn_m))

    df_beta = (df_corr * df_std).apply(lambda x: x * sr_std_m)

    df_pos = df_beta.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                           int(N / n)] else 0 for i in x], axis=1)

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.1.4—因子配置—信用等级利差
def taa_214(df_close, df_cs, k , h, n):
    arr_lclose = np.log(df_close).values
    arr_cs = df_cs.values
    N = df_close.shape[1]
    
    def line_forecast(x, y):
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        return reg.predict(x[-1])[0][0]
        
        
    arr_forecast = np.zeros((df_close.shape[0]-k, df_close.shape[1]))
    for i in range(k, len(df_close)):
        for j in range(df_close.shape[1]):
            arr_forecast[i-k, j] = line_forecast(arr_lclose[i-k:i, j], arr_cs[i-k:i, j])
            
    df_forecast = pd.DataFrame(arr_forecast, index=df_close.index[k:])

    df_undervalue = df_forecast / np.log(df_close).shift(1)

    df_pos = df_undervalue.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                           int(N / n)] else 0 for i in x], axis=1)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value





# 策略2.1.5—因子配置—Max

def taa_215(df_rtn, k, h, n):
    N = df_rtn.shape[1]
    df_max = df_rtn.rolling(k).max()
    df_pos = df_max.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                          int(N / n)] else 0 for i in x], axis=1)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.1.6—因子配置—高阶矩

def taa_216(df_rtn, k, h, n):
    N = df.shape[1]
    df_skew = df_rtn.rolling(k).skew()
    df_kurt = df_rtn.rolling(k).kurt()

    df_pos1 = df_skew.apply(lambda x: [1 if i > sorted(x, reverse=False)[
                            int(N / n)] else 0 for i in x], axis=1)
    df_pos2 = df_kurt.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                            int(N / n)] else 0 for i in x], axis=1)

    df_pos = df_pos1 * df_pos2

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.1.7—因子配置—下跌风险

def taa_217(df_rtn, sr_rtn_m, k, h, n):
    N = df.shape[1]
    y_plus = sr_rtn_m[sr_rtn_m > 0]
    x_plus = df_rtn[sr_rtn_m > 0]
    y_minus = sr_rtn_m[sr_rtn_m < 0]
    x_minus = df_rtn[sr_rtn_m < 0]

    beta_plus = y_plus.rolling(k).apply(
        lambda x: x.cov(x_plus)) / x_plus.rolling(k).var()
    beta_minus = y_minus.rolling(k).apply(
        lambda x: x.cov(x_minus)) / x_minus.rolling(k).var()

    df_pos = (beta_plus - beta_minus).apply(lambda x: [1 if i > sorted(
        x, reverse=True)[int(N / n)] else 0 for i in x], axis=1)

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.1.8—因子配置—非系统风险

def taa_218(df_rtn, sr_rtn_m, k, h, n):
    N = df.shape[1]
    r = df_rtn.values
    rm = sr_rtn_m.values
    arr_rsd = np.array((df_rtn.shape[0], df_rtn.shape[1]))
    for i in range(k, len(df_rtn)+1):
        rsd = np.linalg.lstsq(
            np.vstack([rm[i-k:i], np.ones(k)]).T, r[i-k:i, :], rcond=-1)[1]
        arr_rsd[i] = rsd

    df_pos = pd.DataFrame(arr_rsd, index=df_rtn.index[k-1:]).apply(
        lambda x: [1 if i > sorted(x, reverse=True)[int(N / n)] else 0 for i in x], axis=1)

    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略2.2.1—因子择时—RORO指标

def taa_221(df_rtn, k, h):
    def first_variance(X):
        pca = PCA(n_components=1)
        pca.fit(X)
        return pca.explained_variance_

    l_var = []
    for i in range(k, len(df_rtn)+1):
        X = df_rtn.iloc[i-k:i, :]
        l_var.append(first_variance(X))
    sr_RORO = pd.Series(l_var, index=df_rtn.index[k-1:])

    return sr_RORO



# 策略2.2.2—因子择时—吸收比率指标

def taa_222(df_rtn, k1, k2, h, first_n):
    def ar_n(X, n):
        pca = PCA(n_components=n)
        pca.fit(X)
        return pca.explained_variance_ratio.sum()

    l_ar_1 = []
    l_ar_2 = []
    for i in range(k2, len(df_rtn)+1):
        X_1 = df_rtn.iloc[i-k1:i, :]
        X_2 = df_rtn.iloc[i-k2:i, :]
        l_ar_1.append(first_variance(X_1))
        l_ar_2.append(first_variance(X_2))

    sr_ar_1 = pd.Series(l_var, index=df_rtn.index[k2-1:])
    sr_ar_2 = pd.Series(l_var, index=df_rtn.index[k2-1:])

    return sr_ar_1 - sr_ar_2


# 策略2.2.3—因子择时—波动率指标+SMA

def taa_223(df_rtn, k, k1, k2, sma, h):
    df_std = df_rtn.rolling(k).std()
    df_std_ma1 = df_std.rolling(k1).mean()
    df_std_ma2 = df_std.rolling(k2).mean()

    df_pos = np.where(df_std_ma1 > df_std_ma2, 1, 0)
    sr_value = pos2value(df, df_pos, h)
    return sr_value



# 策略2.2.4—因子择时—流动性指标

def taa_224(df_rtn, df_amt, threshold, k, h):
    amihud = (df_rtn.abs() / df_amt).rolling(k).mean().applymap(np.log)
    # TODO hp func
    amihud_hp = amihud.expanding(min_periods=100).apply(hp)
    df_pos = amihud_hp.applymap(lambda x: 1 if x > threshold else 0)

    df_pos = np.where(df_std_ma1 > df_std_ma2, 1, 0)
    sr_value = pos2value(df, df_pos, h)
    return sr_value



# 策略4.1—目标波动率—标准CVo

def taa_410(df_rtn, target_sigma, k, h):
    df_u = df_rtn.rolling(k).apply(lambda x: (1 + x).prod() - 1)
    df_cov = df_rtn.rolling(k).cov()

    l_date = df_rtn.index.tolist()
    l_w = []
    for date in l_date:
        u = df_u.loc[date, :]
        cov = df_cov.loc[date, :]
        l_w.append(portfolio.target_vol(u, cov, target_sigma))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value


# 策略5.2.1—系统beta—最大去相关性DeCorr

def taa_521(df_rtn, k, h):

    df_corr = df_rtn.rolling(k).corr()
    N = df_rtn.shape[1]

    def optim(rho):

        def func(x, rho, sign=1.0):
            res = x.dot(rho).dot(x)
            return sign * res

        cons = ({'type': 'eq',
                 'fun': lambda x: np.array(x.sum() - 1.0),
                 'jac': lambda x: np.ones(N)})

        res = minimize(func, [1/N]*N, args=(rho),
                       constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

        return res.x

    l_w = []
    for gp in df_corr.groupby(level=0):
        l_w.append(optim(gp[1]))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略5.2.2—系统beta—最大夏普比率Sharpe

def taa_522(df_rtn, k, h):
    df_u = df_rtn.rolling(k).apply(lambda x: (1 + x).prod() - 1)
    df_cov = df_rtn.rolling(k).cov()
    N = df_rtn.shape[1]

    def optim(u, cov):

        def func(x, u, cov, sign=-1.0):
            res = x.dot(u) / (x.dot(cov).dot(x)) ** 0.5
            return sign * res

        cons = ({'type': 'eq',
                 'fun': lambda x: np.array(x.sum() - 1.0),
                 'jac': lambda x: np.ones(N)})

        res = minimize(func, [1/N]*N, args=(u, rho),
                       constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

        return res.x

    l_date = df_rtn.index.tolist()
    l_w = []
    for date in l_date:
        u = df_u.loc[date, :]
        cov = df_cov.loc[date, :]
        l_w.append(optim(u, cov))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略5.2.3—系统beta—最小波动率MinVol

def taa_523(df_rtn, k, h):
    df_cov = df_rtn.rolling(k).cov()
    N = df_rtn.shape[1]

    l_w = []
    for gp in df_corr.groupby(level=0):
        l_w.append(portfolio.min_vol(gp[1]))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略5.2.4—系统beta—风险平价RP

def taa_524(df_rtn, k, h):
    df_cov = df_rtn.rolling(k).cov()
    N = df_rtn.shape[1]

    l_w = []
    for gp in df_cov.groupby(level=0):
        l_w.append(portfolio.rp(gp[1]))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略5.2.5—系统beta—最大分散化MD

def taa_525(df_rtn, k, h):
    df_cov = df_rtn.rolling(k).cov()
    df_std = df_rtn.rolling(k).std()
    N = df_rtn.shape[1]

    def optim(std, cov):

        def func(x, std, cov, sign=-1.0):
            res = x.dot(std) / (x.dot(cov).dot(x)) ** 0.5
            return sign * res

        cons = ({'type': 'eq',
                 'fun': lambda x: np.array(x.sum() - 1.0),
                 'jac': lambda x: np.ones(N)})

        res = minimize(func, [1/N]*N, args=(std, cov),
                       constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

        return res.x

    l_date = df_rtn.index.tolist()
    l_w = []
    for date in l_date:
        std = df_std.loc[date, :]
        cov = df_cov.loc[date, :]
        l_w.append(optim(std, cov))

    df_pos = pd.DataFrame(l_w, index=df_rtn.index, columns=df_rtn.columns)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value



# 策略5.3.1—组合策略—XSM+RP+SMA

def taa_531(df_rtn, k1, k2, k3, h, n):
    df_tsm = df_rtn.rolling(k1).apply(lambda x: (1 + x).prod() - 1)
    df_cov = df_rtn.rolling(k2).cov()
    #df_ma = df_rtn.rolling(k3).mean()
    N = df.shape[1]

    df_pos = df_rtn.apply(lambda x: [1 if i > sorted(x, reverse=True)[
                          int(N / n)] else 0 for i in x], axis=1)

    l_date = df_rtn.index.tolist()
    df_w = pd.DataFrame()
    for date in l_date:
        cov = df_cov.loc[date, :]
        pos = df_pos.loc[date, :]
        cov = cov.loc[pos > 0, pos > 0]
        sr_w = pd.Series(portfolio.rp(cov), index=cov.index)
        df_w = pd.concat([df_w, sr_w], axis=1)

    df_pos = df_w.T.reindex(index=df_rtn.index)
    sr_value = pos2value(df_rtn, df_pos, h)

    return sr_value
