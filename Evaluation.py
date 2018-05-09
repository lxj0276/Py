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
            ret_bench = self.ret_p.copy()
            ret_bench[:] = 0

    def _ret_p(self):
        ret = (df_Pos * df_ret).sum(axis=1)
        ret_annual = ret.apply(lambda x: (1 + x) ** (250 / self.delta_days))
        return ret_annual

    def _net_value(self):
        sr_turnover = self.pos.diff(1).abs().sum(axis=1) / 2.0
        return (1 + (self.pos * self.ret).sum(axis=1) - sr_turnover * trade_cost).cumpord()

    def _trade_days(self):
        timedelta = (self.pos.index[-1] - self.pos.index[0]).days
        return timedelta * 250.0 / 360.0

    def _delta_days(self):
        return round(self.trade_days / len(self.pos))

    def Weight(self):
        return self.pos

    def Avg_Weight(self):
        return self.pos.mean(axis=1)

    def RetC(self):
        return (self.pos * self.ret).apply(lambda x: x / x.sum(), axis=1).mean()

    def RiskC(self):
        pass

    def Turnover(self):
        return self.pos.diff(1).abs().sum().sum() / 2.0 * 250.0 / self.trade_days

    def _Ret_Roll(self, Y):
        lag = int(Y * 250 / self.delta_days)
        r1 = self.value.pct_change(lag).mean()
        r2 = self.value.pct_change(lag).max()
        r3 = self.value.pct_change(lag).min()
        return r1, r2, r3

    def Ret_Roll_1Y(self):
        return self._Ret_Roll(1)

    def Ret_Roll_3Y(self):
        return self._Ret_Roll(3)

    def Ret_Roll_5Y(self):
        return self._Ret_Roll(5)

    def YearRet(self):
        return self.net_value.groupby(lambda x: x[:4]).agg(lambda x: x[-1] / x[0] - 1)

    def YearExRet(self):
        return self.YearRet() - rate_riskfree

    def Stdev(self):
        return self.ret_p.std() * (250 / self.delta_days) ** 0.5

    def _part_Stdev(l, l_part):
        part_var = np.var(l_part) * len(l_part) / len(l)
        return part_var ** 0.5

    def U_Stdev(self):
        U_rtn_p = self.ret_p[self.ret_p > 0]
        U_stdev = self._part_Stdev(
            self.ret_p, U_ret_p) * (250 / self.delta_days) ** 0.5
        return U_stdev

    def D_Stdev(self):
        D_rtn_p = self.ret_p[self.ret_p < 0]
        D_stdev = self._part_Stdev(
            self.ret_p, D_ret_p) * (250 / self.delta_days) ** 0.5
        return D_stdev

    def Skew(self):
        return self.ret_p.skew()

    def Kurt(self):
        return self.ret_p.kurt()

    def VaR(self, q=0.05):
        return self.ret_p.apply(lambda x: (1 + x) ** (250 / self.delta_days) - 1).quantile(q)

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
        return self.ret_p.corr(x)

    def JAlpha(self, ret_bench):
        X = np.vstack([ret_bench.values, np.ones(len(ret_bench))])
        _, c = np.linalg.lstsq(X, self.ret_p.values -
                               rate_riskfree, rcond=None)[0]
        return c

    def TM(self, ret_bench):
        x1 = ret_bench.values
        x2 = x1 ** 2
        X = np.vstack([x1, x2, np.ones(len(ret_bench))])
        _, c, _ = np.linalg.lstsq(
            X, self.ret_p.values - rate_riskfree, rcond=None)[0]
        return c

    def HM(self, ret_bench):
        x1 = ret_bench.values
        x2 = np.where(x1 > 0, x1, 0)
        X = np.vstack([x1, x2, np.ones(len(ret_bench))])
        _, c, _ = np.linalg.lstsq(
            X, self.ret_p.values - rate_riskfree, rcond=None)[0]
        return c

    def SR(self):
        return (self.ret_p.mean() - rate_riskfree) / self.ret_p.std()

    def ASSR(self):
        SR = self.SR()
        return SR * (1 + self.Skew() * SR / 6.0)

    def ASKSR(self):
        SR = self.SR()
        return SR * (1 + self.Skew() * SR / 6.0 - self.Kurt() * SR ** 2 / 24.0)

    def DSR(self):
        pass

    def Treynor(self, ret_bench):
        X = np.vstack([ret_bench.values, np.ones(len(ret_bench))])
        beta, _ = np.linalg.lstsq(
            X, self.ret_p.values - rate_riskfree, rcond=None)[0]
        return (self.ret_p.mean() - rate_riskfree) / beta

    def Sortino(self, MAR=0):
        D_ret_p = self.ret_p[self.ret_p < MAR]
        DR = self._part_Stdev(self.ret_p, D_ret_p)
        return (self.ret_p - MAR) / DR

    def Calmar(self):
        return (self.ret_p.mean() - rate_riskfree) / self.MaxDD()

    def PainRatio(self):
        return (self.ret_p.mean() - rate_riskfree) / self.PainInd()

    def RoVap(self):
        return (self.ret_p.mean() - rate_riskfree) / self.VaR()

    def Hit_Rate(self, ret_bench):
        return np.where(self.ret_p > ret_bench, 1, 0).mean()

    def Gain2Pain(self):
        ret = self.ret_p
        return ret[ret > 0].sum() / ret[ret < 0].sum()

    def IR(self, ret_bench):
        track_error = self.ret_p - ret_bench
        IR = track_error.mean() / track_error.std()
        return IR
