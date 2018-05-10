# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:25:05 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd

class evaluation(object):

    def __init__(self, df_pos, df_ret, ret_bench=None, trade_cost=0, rate_riskfree=0):
        self.pos = df_pos
        self.ret = df_ret
        self.tc = trade_cost
        self.rrf = rate_riskfree
        self.ret_p = self._ret_p()
        self.net_value = self._net_value()
        self.trade_days = self._trade_days()
        self.delta_days = self._delta_days()
        if not ret_bench:
            self.ret_bench = self.ret_p.copy()
            self.ret_bench[:] = 0

    def _ret_p(self):
        ret = (self.pos * self.ret).sum(axis=1)
        return ret

    def _net_value(self):
        sr_turnover = self.pos.diff(1).abs().sum(axis=1) / 2.0
        return (1 + self.ret_p - sr_turnover * self.tc).cumprod().dropna()

    def _trade_days(self):
        timedelta = (self.pos.index[-1] - self.pos.index[0]).days
        return round(timedelta * 250.0 / 360.0)

    def _delta_days(self):
        return round(self.trade_days / len(self.pos))
    
    def _ret_annual(self, x):
        return (1 + x) ** (250 / self.delta_days) - 1
    
    def _vol_annual(self, x):
        return x * (250 / self.delta_days) ** 0.5
    
    def Weight(self):
        return self.pos

    def Avg_Weight(self):
        return self.pos.mean()

    def RetC(self):
        return (self.pos * self.ret).apply(lambda x: x / x.sum(), axis=1).mean()
    
    def _RC(self, w, Sigma):
        return (w * Sigma.dot(w)) / (w.dot(Sigma).dot(w) ** 0.5)
    
    def RiskC(self):
        cov = self.ret.ewm(alpha=0.8).cov()
        df_rc = self.pos.copy()
        for index, rows in self.pos.iterrows():
            rc = self._RC(rows, cov.loc[index, :])
            df_rc.loc[index, :] = rc
        return df_rc.mean()

    def Turnover(self):
        return self.pos.diff(1).abs().sum().sum() / 2.0 * 250.0 / self.trade_days

    def Average(self):
        return self._ret_annual(self.ret_p.mean())

    def CAGR(self):
        return (self.net_value[-1] / self.net_value[0]) ** (250.0 / self.trade_days) - 1.0

    def _Ret_Roll(self, Y):
        lag = int(Y * 250 / self.delta_days)
        ret_roll = self.net_value.rolling(lag).apply(lambda x: x[-1] / x[0] ** (1.0 / Y) - 1.0)
        r_mean, r_max, r_min = ret_roll.mean(), ret_roll.max(), ret_roll.min()
        return r_mean, r_max, r_min

    def Ret_Roll_1Y(self):
        return self._Ret_Roll(1)

    def Ret_Roll_3Y(self):
        return self._Ret_Roll(3)

    def Ret_Roll_5Y(self):
        return self._Ret_Roll(5)

    def YearRet(self):
        return self.net_value.groupby(lambda x: x.year).apply(lambda x: x[-1] / x[0] - 1)

    def YearExRet(self):
        return self.YearRet() - self.rrf

    def CAGR_Bull_Bear(self, sr_state):
        sr_state.name = "state"
        tmp = pd.concat([self.ret_p, sr_state], axis=1)
        return tmp.groupby(by="state").agg(np.mean).apply(self._ret_annual)

    def CAGR_Factor(self, sr_factor):
        sr_factor.name = "factor"
        tmp = pd.concat([self.ret_p, sr_factor], axis=1)
        return tmp.groupby(by="factor").agg(np.mean).apply(self._ret_annual)

    def ExCAGR_Bull_Bear(self, sr_state):
        sr_state.name = "state"
        tmp = pd.concat([self.ret_p, sr_state], axis=1)
        return (tmp.groupby(by="state").agg(np.mean) - self.rrf).apply(self._ret_annual)

    def ExCAGR_Factor(self, sr_factor):
        sr_factor.name = "factor"
        tmp = pd.concat([self.ret_p, sr_factor], axis=1)
        return (tmp.groupby(by="factor").agg(np.mean) - self.rrf).apply(self._ret_annual)

    def Stdev(self):
        return self._vol_annual(self.ret_p.std())

    def _part_Stdev(self, l, l_part):
        part_var = np.var(l_part) * len(l_part) / len(l)
        return part_var ** 0.5

    def U_Stdev(self):
        U_ret_p = self.ret_p[self.ret_p > 0]
        U_stdev = self._vol_annual(self._part_Stdev(self.ret_p, U_ret_p))
        return U_stdev

    def D_Stdev(self):
        D_ret_p = self.ret_p[self.ret_p < 0]
        D_stdev = self._vol_annual(self._part_Stdev(self.ret_p, D_ret_p))
        return D_stdev

    def Skew(self):
        return self.ret_p.skew()

    def Kurt(self):
        return self.ret_p.kurt()

    def VaR(self, q=0.05):
        return -1 * self.ret_p.apply(self._ret_annual).quantile(q)

    def DD(self):
        return self.net_value.expanding().apply(lambda x: 1 - x[-1] / x.max())

    def MaxDD(self):
        return self.DD().max()

    def DD_Dur(self):
        return self.net_value.expanding().apply(lambda x: 1 if x[-1] < x.max() else np.nan)

    def MaxDD_Dur(self):
        return max([self.DD_Dur().rolling(i).sum().max() for i in range(len(self.DD_Dur()))])

    def PainInd(self):
        return -1 * self.DD_Dur().sum() / len(self.DD_Dur())

    def corr(self, X):
        return self.ret_p.corr(X)

    def JAlpha(self):
        X = np.vstack([self.ret_bench.values, np.ones(len(self.ret_bench))])
        _, c = np.linalg.lstsq(X.T, self.ret_p.values, rcond=None)[0]
        return c

    def TM(self):
        x1 = self.ret_bench.values
        x2 = x1 ** 2
        X = np.vstack([x1, x2, np.ones(len(self.ret_bench))])
        _, c, _ = np.linalg.lstsq(X.T, self.ret_p.values, rcond=None)[0]
        return c

    def HM(self):
        x1 = self.ret_bench.values
        x2 = np.where(x1 > 0, x1, 0)
        X = np.vstack([x1, x2, np.ones(len(self.ret_bench))])
        _, c, _ = np.linalg.lstsq(X.T, self.ret_p.values, rcond=None)[0]
        return c

    def SR(self):
        return (self.CAGR() - self.rrf) / self.Stdev()

    def ASSR(self):
        SR = self.SR()
        return SR * (1 + self.Skew() * SR / 6.0)

    def ASKSR(self):
        SR = self.SR()
        return SR * (1 + self.Skew() * SR / 6.0 - self.Kurt() * SR ** 2 / 24.0)

    def DSR(self):
        SR_roll = (self.ret_p.rolling(int(250 / self.delta_days)).mean() -
                   self.rrf) / self.ret_p.rolling(int(250 / self.delta_days)).std()
        return SR_roll.mean() / SR_roll.std()

    def Treynor(self):
        X = np.vstack([self.ret_bench.values, np.ones(len(self.ret_bench))])
        beta, _ = np.linalg.lstsq(
            X.T, self.ret_p.values, rcond=None)[0]
        return self.CAGR() / beta

    def Sortino(self, MAR=0):
        D_ret_p = self.ret_p[self.ret_p < MAR]
        DR = self._part_Stdev(self.ret_p, D_ret_p)
        return (self.ret_p.mean() - MAR) / DR

    def Calmar(self):
        return (self.ret_p.mean() - self.rrf) / self.MaxDD()

    def PainRatio(self):
        return (self.ret_p.mean() - self.rrf) / self.PainInd()

    def RoVap(self):
        return (self.ret_p.mean() - self.rrf) / self.VaR()

    def Hit_Rate(self):
        return np.where(self.ret_p > self.ret_bench, 1, 0).mean()

    def Gain2Pain(self):
        ret = self.ret_p
        return ret[ret > 0].sum() / (-1 * ret[ret < 0].sum())

    def IR(self):
        track_error = self.ret_p - self.ret_bench
        IR = track_error.mean() / track_error.std()
        return IR
#######################################################################################

# Demo
from strategy_beta import weights_solver

if __name__ == "__main__":

    # 读收益率数据
    df_rtn = pd.read_csv('rtn.csv', index_col=0)

    # 收益率和协方差的预测
    rtn_p = df_rtn.rolling(20).mean().shift(1)
    cov_p = df_rtn.rolling(20).cov().shift(1)

    # 转换为list
    l_month = rtn_p.dropna().index.tolist()
    l_r = []
    l_Sigma = []
    for m in l_month:
        u = np.array(rtn_p.loc[m])
        Sigma = np.array(cov_p.loc[m])
        l_r.append(u)
        l_Sigma.append(Sigma)

    # 用最大夏普比优化
    l_weights = weights_solver("risk_parity", l_Sigma)

    # 评价指标
    df_pos = pd.DataFrame(l_weights, columns=df_rtn.columns, index=df_rtn.index[21:])
    E = evaluation(df_pos, df_rtn[21:])
    E.net_value
    E.CAGR()
    E.IR()
    # ...