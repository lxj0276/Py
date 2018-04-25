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
