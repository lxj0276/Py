def clear(arr, d=2):
    brr = arr.copy()
    for i in range(len(arr)):
        brr[i] = round(arr[i], d)
    brr[brr.argmax()] = brr[brr.argmax()] + (1.0 - brr.sum())

    return brr

# z为净值list，f=20代表数据采集间隔为20个交易日


def indics(z, f=20):
    N = len(z)
    rtn = (z[-1] / z[0]) ** (250 / f / N) - 1
    vol = z.pct_change().std() * ((250 / f) ** 0.5)
    sp = (rtn - 0.04) / vol
    dd = []
    for i in range(N):
        Z = z[:i+1]
        dd.append(1 - Z[-1] / Z.max())
    dd_max = max(dd)
    cm = rtn / dd_max

    return rtn, vol, sp, dd_max, cm

# -*- coding: utf-8 -*-

import numpy as np


def file2list(file):
    with open(file) as f:
        List = f.read().split()
    return List


def countTdays(startDate, endDate, tDayFile="tradedate.csv"):
    tDayList = file2list(tDayFile)
    tDays = tDayList.index(endDate) - tDayList.index(startDate)
    return tDays


class indics(object):
    """docstring for indics"""

    def __init__(self, sr_value):
        self.value = sr_value.values
        self._sr = sr_value
        dateList = sr_value.index.tolist()
        self._freq = len(dateList)
        self._tDays = countTdays(dateList[0].strftime(
            "%Y-%m-%d"), dateList[-1].strftime("%Y-%m-%d"))

    @property
    def rtn(self):
        return (self.value[-1] / self.value[0]) ** (250 / self._tDays) - 1

    @property
    def vol(self):
        return self._sr.pct_change(1).std() * ((250 / (self._tDays / self._freq)) ** 0.5)

    @property
    def sp(self):
        return (self.rtn - 0.04) / self.vol

    @property
    def dd_max(self):
        dd = []
        for i in range(len(self.value)):
            Z = self.value[:i+1]
            dd.append(1 - Z[-1] / Z.max())
        return max(dd)

    @property
    def cm(self):
        return self.rtn / self.dd_max
