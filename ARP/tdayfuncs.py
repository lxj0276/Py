# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:49:26 2018

@author: zhangyw49
"""
from datetime import datetime
import pandas as pd

# 交易日期文件
trade_days_file = "./date/trade_Days.csv"

# 从文件读取序列
def file2list(file):
    with open(file) as f:
        List = f.read().split()
    return List

# 读取交易日期序列
def get_trade_days(file=trade_days_file):
    trade_days = file2list(file)
    return trade_days

# 得到今日日期
def get_today():
    today=datetime.today().strftime("%Y-%m-%d")
    return today

# 判断是否为交易日
def is_trade_day(datestr):
    tdays = get_trade_days()
    return (datestr in tdays)

# 日期推移
def tday_shift(datestr, shift_days):
    tdays = get_trade_days()
    if shift_days < 0:
        early_days = [day for day in tdays if day < datestr]
        date = early_days[shift_days]
    elif shift_days > 0:
        early_days = [day for day in tdays if day > datestr]
        date = early_days[shift_days]      
    return date

# 获取期间交易日序列
def get_tday_list(begin_datestr, end_datestr):
    tdays = get_trade_days()
    tday_list = [day for day in tdays if begin_datestr <= day  <= end_datestr]
    return tday_list

# 统计期间交易天数
def count_tday(begin_datestr, end_datestr):
    tdays = get_trade_days()
    tday_list = get_tday_list(begin_datestr, end_datestr)
    begin_idx = tdays.index(tday_list[0])
    end_idx = tdays.index(tday_list[-1])
    num_days = end_idx - begin_idx
    return num_days

# 得到文件最后一行的日期index
def get_latest_day(file):
    L = file2list(file)
    return L[-1][:10]

# 定义换仓时间，本例是从2008年12月31日起，每隔5个交易日进行换仓
def get_order_days():
    trade_days = pd.to_datetime(get_trade_days()).to_series(name="Date")
    start, end = "2008-12-31", tday_shift(get_today(), -1)
    trade_days = trade_days[start:end]
    order_days = trade_days[::5]
    return order_days