# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:49:26 2018

@author: zhangyw49
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


from WindPy import w
w.start()

def get_wsd(raw_wsd):
    df = pd.DataFrame(raw_wsd.Data).T
    df.index = raw_wsd.Times
    return df

def get_multi_wsd(codeList, fields, beginTime, endTime, options=""):
    df = pd.DataFrame()
    for code in codeList:
        raw_wsd = w.wsd(code, fields, beginTime, endTime, options)
        df_tmp = get_wsd(raw_wsd)
        df = pd.concat([df, df_tmp], axis=1)
        
    col_level_1 = [code for code in codeList for _ in fields.split()]
    col_level_2 = fields.split() * len(codeList)
    df.columns=[col_level_1, col_level_2]
    
    return df
'''
stock_size_index = ['000016.SH', '000300.SH', '000852.SH', '000905.SH']
stock_size = get_multi_wsd(stock_size_index, "close", "2000-01-01", "2018-06-30")

stock_sector_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801080.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI',
                      '801180.SI', '801200.SI', '801210.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI', '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI', '801880.SI', '801890.SI']
stock_sector = get_multi_wsd(stock_sector_index, "close", "2000-01-01", "2018-06-30")
'''

bond_treasury_index = ['CBA00621.CS', 'CBA00631.CS', 'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS']
bond_finance_index = ['CBA01221.CS', 'CBA01231.CS', 'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS']
bond_corporate_index = ['CBA04221.CS', 'CBA04231.CS', 'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS']

bond_treasury = get_multi_wsd(bond_treasury_index, "close", "2000-01-01", "2018-06-30")
bond_finance = get_multi_wsd(bond_finance_index, "close", "2000-01-01", "2018-06-30")
bond_corporate = get_multi_wsd(bond_corporate_index, "close", "2000-01-01", "2018-06-30")

