# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:03:47 2018

@author: zhangyw49
"""
import os
import pandas as pd
import tdayfuncs as tdf




def file_to_frame(file):
    df = pd.read_csv(file, index_col=[0], header=[0, 1], parse_dates=[0])
    return df["2008-12-31":]

def get_order_days():
    trade_days = pd.to_datetime(tdf.get_trade_days()).to_series(name="Date")
    start, end = "2008-12-31", tdf.tday_shift(tdf.get_today(), -1)
    trade_days = trade_days[start:end]
    order_days = trade_days[::5]
    return order_days

def position_to_return(price, position, order_days, trade_cost=None):
    pos_order = position.reindex(order_days.index)
    pos = pos_order.reindex(price.index).fillna(method="ffill")
    rtn = price.pct_change(1)
    if not trade_cost:
        rtn_sum = (pos.shift(1) * rtn).sum(axis=1)
    else:
        rtn_after = rtn - pos.diff(1).shift(1).abs().mul(trade_cost/2, axis=1)
        rtn_sum = (pos.shift(1) * rtn_after).sum(axis=1)
    return rtn_sum.dropna()


def compute_return():
    path_single = "./out/single/"
    file_names = os.listdir(path_single+"price/")
    order_days = get_order_days()
    for f in file_names:
        price = file_to_frame(path_single+"price/"+f)
        position = file_to_frame(path_single+"position/"+f)
        rtn = position_to_return(price, position, order_days)
        rtn.to_csv(path_single+"return/%s"%f)
    print("\nReturn Computation Done!")
    return None


if __name__ == "__main__":
    compute_return()


