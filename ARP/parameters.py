# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:24:40 2018

@author: zhangyw49
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 股票规模指数：上证50 沪深300 中证500 中证1000
stock_size_index = ['000016.SH', '000300.SH', '000852.SH', '000905.SH']
# 股票行业申万指数：农林牧渔 采掘 化工 钢铁 有色金属 电子 家用电器 食品饮料 纺织服装 轻工制造 医药生物 公用事业 交通运输 房地产商业贸易 休闲服务 综合 建筑材料 建筑装饰 电气设备 国防军工 计算机传媒 通信 银行 非银金融 汽车 机械设备
stock_sector_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801080.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI',
                      '801180.SI', '801200.SI', '801210.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI', '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI', '801880.SI', '801890.SI']

# 国债财富指数：中债-国债总财富(1-3年)指数   中债-国债总财富(3-5年)指数    中债-国债总财富(5-7年)指数    中债-国债总财富(7-10年)指数   中债-国债总财富(10年以上)指数
bond_treasury_index = ['CBA00621.CS', 'CBA00631.CS', 'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS']
# 金融债财富指数：中债-金融债券总财富(1-3年)指数   中债-金融债券总财富(3-5年)指数  中债-金融债券总财富(5-7年)指数  中债-金融债券总财富(7-10年)指数 中债-金融债券总财富(10年以上)指数
bond_finance_index = ['CBA01221.CS', 'CBA01231.CS', 'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS']
# AAA企业债财富指数：中债-企业债AAA财富(1-3年)指数   中债-企业债AAA财富(3-5年)指数 中债-企业债AAA财富(5-7年)指数 中债-企业债AAA财富(7-10年)指数    中债-企业债AAA财富(10年以上)指数
bond_corporate_index = ['CBA04221.CS', 'CBA04231.CS', 'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS']

# 跨类指数：中证全指  恒生中国企业指数  中债-国债总财富(总值)指数 中债-金融债券总财富(总值)指数 中债-企业债AAA财富(总值)指数
asset_class_index = ['000985.CSI', 'HSCEI.HI', 'CBA00601.CS', 'CBA01201.CS', 'CBA04201.CS']



param_stock_size_mom = {"m": 5, "lag": [210, 240, 270], "position": [0.5, 0.5, -0.5, -0.5]}
param_stock_size_vol = {"m": 5, "lag": [240, 270, 300], "position": [0.5, 0.5, -0.5, -0.5]}

param_stock_sector_mom = {"m": 5, "lag": [210, 240, 270], "position": [1/14]*14 + [-1/14]*14}
param_stock_sector_vol = {"m": 5, "lag": [240, 270, 300], "position": [1/14]*14 + [-1/14]*14}


param_bond_treasury_csm = {"m": 5, "lag": [240, 270, 300], "position": [1/14]*14 + [-1/14]*14}







# 交易日期文件
trade_days_file = "C:/Users/zhangyw49/code/py/tDays.csv"
df_date = pd.read_csv(trade_days_file, parse_dates=[0], index_col=[0], header=[0])
order_date = df_date["20081231":"20180531"].iloc[::5, :]
# 由order_date控制交易信号


## 计算因子
stock_size = pd.read_csv('stock_size.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_size_mom = stock_mom(stock_size)
stock_size_vol = stock_vol(stock_size)

stock_sector = pd.read_csv('stock_sector.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_sector_mom = stock_mom(stock_sector)
stock_sector_vol = stock_vol(stock_sector)


## 计算净值
sr_rtn = net_value(stock_size, stock_size_mom)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_size, stock_size_vol)
(sr_rtn + 1).cumprod().plot()

sr_rtn = net_value(stock_sector, stock_sector_mom, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_sector, stock_sector_vol, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()
