# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:24:40 2018

@author: zhangyw49
"""

code_dict = {
# 股票规模指数：上证50 沪深300 中证500 中证1000
"stock_size" : ['000016.SH', '000300.SH', '000852.SH', '000905.SH'],
# 股票行业指数(申万一级)：农林牧渔 采掘 化工 钢铁 有色金属 电子 家用电器 食品饮料 
# 纺织服装 轻工制造 医药生物 公用事业 交通运输 房地产商业贸易 休闲服务 综合 
# 建筑材料 建筑装饰 电气设备 国防军工 计算机传媒 通信 银行 非银金融 汽车 机械设备
"stock_sector" : ['801010.SI', '801020.SI', '801030.SI', '801040.SI', 
                  '801050.SI', '801080.SI', '801110.SI', '801120.SI', 
                  '801130.SI', '801140.SI', '801150.SI', '801160.SI', 
                  '801170.SI', '801180.SI', '801200.SI', '801210.SI', 
                  '801230.SI', '801710.SI', '801720.SI', '801730.SI', 
                  '801740.SI', '801750.SI', '801760.SI', '801770.SI', 
                  '801780.SI', '801790.SI', '801880.SI', '801890.SI'],

# 国债财富指数：中债-国债总财富(1-3年)指数   中债-国债总财富(3-5年)指数    
# 中债-国债总财富(5-7年)指数    中债-国债总财富(7-10年)指数   中债-国债总财富(10年以上)指数
"bond_treasury" : ['CBA00621.CS', 'CBA00631.CS', 'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS'],
# 金融债财富指数：中债-金融债券总财富(1-3年)指数   中债-金融债券总财富(3-5年)指数  
# 中债-金融债券总财富(5-7年)指数  中债-金融债券总财富(7-10年)指数 中债-金融债券总财富(10年以上)指数
"bond_finance" : ['CBA01221.CS', 'CBA01231.CS', 'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS'],
# AAA企业债财富指数：中债-企业债AAA财富(1-3年)指数   中债-企业债AAA财富(3-5年)指数 
# 中债-企业债AAA财富(5-7年)指数 中债-企业债AAA财富(7-10年)指数    中债-企业债AAA财富(10年以上)指数
"bond_corporate" : ['CBA04221.CS', 'CBA04231.CS', 'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS'],

# 跨类指数：中证全指  恒生中国企业指数  中债-国债总财富(总值)指数 
# 中债-金融债券总财富(总值)指数 中债-企业债AAA财富(总值)指数
"multi_asset" : ['000985.CSI', 'HSCEI.HI', 'CBA00601.CS', 'CBA01201.CS', 'CBA04201.CS'],

# 无风险利率：DR007
"bond_dr007_rate" : ['DR007.IB'],

# 国债YTM：2年, 4年, 6年, 8年, 9年, 10年, 20年
"bond_treasury_ytm" : ["M1000159", "M1000161", "M1000163", 
                       "M1000165", "M1004678", "M1000166", "M1000168"],
# 国开债YTM：2年, 4年, 6年, 8年, 9年, 10年, 20年
"bond_finance_ytm" : ["M1004264", "M1004266", "M1004268", 
                      "M1004270", "M1004688", "M1004271", "M1004273"],
# 3A企业债YTM：2年, 4年, 6年, 8年, 9年, 10年, 15年
"bond_corporate_ytm" : ["M1000369", "M1000371", "M1000373", 
                        "M1000375", "M1006943", "M1000376", "M1000377"],

# 跨类资产估值
##债券YTM：国债10年   金融债5年   企业债5年
##股票PE、PB、PS
"multi_asset_ytm" : ["M1000162", "M1004267", "M1000372"],
"multi_asset_pe" : ['000985.CSI', 'HSCEI.HI']
}

trade_cost = [0.001] * 32 + [0.0005] * 21

param_dict = {
              "stock": {
                        "size":   {
                                   "mom": {
                                           "m": 5, "lags": [210, 240, 270], "positions": [0.5, 0.5, -0.5, -0.5]
                                          },
                                   "vol": {
                                           "m": 5, "lags": [240, 270, 300], "positions": [0.5, 0.5, -0.5, -0.5]
                                         }
                                  },
                        "sector": {
                                   "mom": {
                                           "m": 5, "lags": [210, 240, 270], "positions": [0.1] * 10 + [0] * 8 + [-0.1] * 10
                                          },
                                   "vol": {
                                           "m": 5, "lags": [240, 270, 300], "positions": [0.1] * 10 + [0] * 8 + [-0.1] * 10
                                          }
                                  }
                       },


              "bond": {
                       "treasury": {
                                    "csm":  {
                                             "lags": [20, 40, 60], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "tsm":  {
                                             "lags": [20, 40, 60], "positions": [-0.5, -0.5, 0, 0.5, 0.5]
                                            },
                                    "value":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]], "m": 5, "lags": [50, 100, 150], 
                                             "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "carry":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]], "x": [7/365, 2, 4, 6, 17/2, 50/3], 
                                             "m": 5, "lags": [50, 100, 150], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            }
                                   },

                        "finance": {
                                    "csm":  {
                                             "lags": [20, 40, 60], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "tsm":  {
                                             "lags": [20, 40, 60], "positions": [-0.5, -0.5, 0, 0.5, 0.5]
                                            },
                                    "value":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]], "m": 5, "lags": [50, 100, 150], 
                                             "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "carry":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]], "x": [7/365, 2, 4, 6, 17/2, 50/3], 
                                            "m": 5, "lags": [50, 100, 150], "positions": [0.5, 0.5, 0, -0.5, -0.5]                                    }
                                   },

                        "corporate": {
                                    "csm":  {
                                             "lags": [20, 40, 60], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "tsm":  {
                                             "lags": [20, 40, 60], "positions": [-0.5, -0.5, 0, 0.5, 0.5]
                                            },
                                    "value":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/2,1/2]], "m": 5, "lags": [50, 100, 150], 
                                             "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            },
                                    "carry":{
                                             "transform_matrix": [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], 
                                             [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/2,1/2]], "x": [7/365, 2, 4, 6, 17/2, 50/3], 
                                             "m": 5, "lags": [50, 100, 150], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                            }
                                      }
                      },

            "multi": {
                        "asset": {
                                   "rev": {
                                           "lags": [240, 300, 360], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                          },
                                  "vol":  {
                                           "lags": [60, 120, 180], "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                          },
                                  "value":{
                                           "m": 5, "positions": [0.5, 0.5, 0, -0.5, -0.5]
                                          }
                                  }
                      }
              }

