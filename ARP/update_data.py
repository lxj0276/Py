# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:24:40 2018

@author: zhangyw49
"""
import numpy as np
import pandas as pd
from WindPy import w

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
        
    col_level_1 = [code for code in codeList for _ in fields.split(sep=',')]
    col_level_2 = fields.split(sep=',') * len(codeList)
    df.columns=[col_level_1, col_level_2]
    return df

def get_edb(codeList, beginTime, endTime, options=""):
    out = w.edb(codeList, beginTime, endTime, options)
    data = np.array(out.Data).T
    index = out.Times
    col_level_1 = codeList
    col_level_2 = out.Fields * len(codeList)
    df = pd.DataFrame(data, index=index, columns=[col_level_1, col_level_2])
    return df

# 股票规模指数：上证50 沪深300 中证500 中证1000
stock_size_index = ['000016.SH', '000300.SH', '000852.SH', '000905.SH']
# 股票行业指数(申万一级)：农林牧渔 采掘 化工 钢铁 有色金属 电子 家用电器 食品饮料 
# 纺织服装 轻工制造 医药生物 公用事业 交通运输 房地产商业贸易 休闲服务 综合 
# 建筑材料 建筑装饰 电气设备 国防军工 计算机传媒 通信 银行 非银金融 汽车 机械设备
stock_sector_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', 
                      '801050.SI', '801080.SI', '801110.SI', '801120.SI', 
                      '801130.SI', '801140.SI', '801150.SI', '801160.SI', 
                      '801170.SI', '801180.SI', '801200.SI', '801210.SI', 
                      '801230.SI', '801710.SI', '801720.SI', '801730.SI', 
                      '801740.SI', '801750.SI', '801760.SI', '801770.SI', 
                      '801780.SI', '801790.SI', '801880.SI', '801890.SI']

# 国债财富指数：中债-国债总财富(1-3年)指数   中债-国债总财富(3-5年)指数    
# 中债-国债总财富(5-7年)指数    中债-国债总财富(7-10年)指数   中债-国债总财富(10年以上)指数
bond_treasury_index = ['CBA00621.CS', 'CBA00631.CS',
                       'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS']
# 金融债财富指数：中债-金融债券总财富(1-3年)指数   中债-金融债券总财富(3-5年)指数  
# 中债-金融债券总财富(5-7年)指数  中债-金融债券总财富(7-10年)指数 中债-金融债券总财富(10年以上)指数
bond_finance_index = ['CBA01221.CS', 'CBA01231.CS',
                      'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS']
# AAA企业债财富指数：中债-企业债AAA财富(1-3年)指数   中债-企业债AAA财富(3-5年)指数 
# 中债-企业债AAA财富(5-7年)指数 中债-企业债AAA财富(7-10年)指数    中债-企业债AAA财富(10年以上)指数
bond_corporate_index = ['CBA04221.CS', 'CBA04231.CS',
                        'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS']

# 跨类指数：中证全指  恒生中国企业指数  中债-国债总财富(总值)指数 
# 中债-金融债券总财富(总值)指数 中债-企业债AAA财富(总值)指数
asset_class_index = ['000985.CSI', 'HSCEI.HI',
                     'CBA00601.CS', 'CBA01201.CS', 'CBA04201.CS']

# 无风险利率：DR007
bond_dr007 = ['DR007.IB']

# 国债YTM：2年, 4年, 6年, 8年, 9年, 10年, 20年
bond_treasury_ytm = ["M1000159", "M1000161", "M1000163", 
                     "M1000165", "M1004678", "M1000166", "M1000168"]
# 国开债YTM：2年, 4年, 6年, 8年, 9年, 10年, 20年
bond_finance_ytm = ["M1004264", "M1004266", "M1004268", 
                    "M1004270", "M1004688", "M1004271", "M1004273"]
# 3A企业债YTM：2年, 4年, 6年, 8年, 9年, 10年, 15年
bond_corporate_ytm = ["M1000369", "M1000371", "M1000373", 
                      "M1000375", "M1006943", "M1000376", "M1000377"]

# 跨类资产估值
##债券YTM：国债10年   金融债5年   企业债5年
##股票PE、PB、PS
asset_value_bond = ["M1000162", "M1004267", "M1000372"]
asset_value_stock = ['000985.CSI', 'HSCEI.HI']


















names_wsd = {"stock_size_index": "stock_size", "stock_sector_index": "stock_sector", 
"bond_treasury_index": "bond_treasury", "bond_finance_index": "bond_finance", 
"bond_corporate_index": "bond_corporate", "bond_dr007":"bond_dr007_rate", 
"asset_class_index": "asset_class"}

names_edb = {"bond_treasury_ytm": "bond_treasury_ytm", "bond_finance_ytm": "bond_finance_ytm", 
"bond_corporate_ytm": "bond_corporate_ytm", "asset_class_ytm": "asset_class_ytm"}

names_other = {}


def update_to_csv(list_name, file_name, fields, type="wsd"):
    df = pd.read_csv('./data/%s.csv'%file_name, index_col=[0], header=[0, 1], parse_dates=[0])
    start_date = df.index[-1].strftime("%Y-%m-%d")
    if type == "wsd":
        df_update = get_multi_wsd(eval(list_name), fields, start_date, None)
    elif type == "edb":
        df_update = get_edb(eval(list_name), start_date, None)
    df_update.iloc[1:, :].to_csv('./data/%s.csv'%file_name, header=False, mode='a')
    return 0


def update_now():
    w.start()
    for list_name, file_name in names_wsd.items():
        update_to_csv(list_name, file_name, "close", type="wsd")
    for list_name, file_name in names_edb.items():
        update_to_csv(list_name, file_name, "close", type="edb")
    update_to_csv(asset_class_index[:2], "asset_value_stock", "pe_ttm,ps_ttm,pb_lf", type="wsd")
    w.stop()


# =============================================================================
# df = pd.read_csv('./data/stock_size.csv', index_col=[0], header=[0, 1], parse_dates=[0])
# start_date = df.index[-1].strftime("%Y-%m-%d")
# df_update = get_multi_wsd(stock_size_index, "close", start_date, None)
# df_update.iloc[1:, :].to_csv('./data/stock_size.csv', header=False, mode='a')
# 
# # 股票
# df_update = get_multi_wsd(stock_size_index, "close", start_date, None)
# df_update = get_multi_wsd(stock_sector_index, "close", start_date, None)
# 
# # 债券
# df_update = get_multi_wsd(bond_treasury_index, "close", start_date, None)
# df_update = get_multi_wsd(bond_finance_index, "close", start_date, None)
# df_update = get_multi_wsd(bond_corporate_index, "close", start_date, None)
# # DR007
# df_update = get_multi_wsd(rate_riskfree, "close", start_date, None)
# 
# # YTM 
# df_update = get_edb(bond_treasury_ytm, start_date, None)
# df_update = get_edb(bond_finance_ytm, start_date, None)
# df_update = get_edb(bond_corporate_ytm, start_date, None)
# 
# # 跨类
# df_update = get_multi_wsd(asset_class_index, "close", start_date, None)
# 
# # 跨类股票
# df_update = get_multi_wsd(asset_class_stock, "pe_ttm,ps_ttm,pb_lf", start_date, None)
# # 跨类债券
# df_update = get_edb(asset_class_bond, start_date, None)
# 
# =============================================================================
