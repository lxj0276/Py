# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:08:51 2018

@author: zhangyw49
"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

df_rtn = pd.read_csv("rtn.csv", index_col=0, parse_dates=[0], header=0, encoding="GBK")

path = "./out/single/return/"
import os
file_names = os.listdir(path)
df_rtn_z = pd.DataFrame()
for f in file_names:
    df = file_to_frame(path + f, header=None)
    df.columns = [f]
    df_rtn_z = pd.concat([df_rtn_z, df], axis=1)

df_value = (1 + df_rtn).cumprod()
df_value_z = (1 + df_rtn_z).cumprod()



df_value_z.loc[:"2017", "stock_sector_vol.csv"].plot()

df_value.loc[:"2017", "股票行业Vol"].plot()
