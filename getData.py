# -*- coding: utf-8 -*-

# get data from Wind
import numpy as np
import pandas as pd
from WindPy import w
from datetime import *


w.start()

def getWsd(wsd):
    df = pd.DataFrame(wsd.Data).T
    df.index = wsd.Times
    df.columns = wsd.Fields 
    return df

def getWset(wset):
    df = pd.DataFrame(wset.Data).T
    df.columns = wset.Fields
    df.set_index('date', inplace=True)
    return df

wsd = w.wsd("000852.SH", "pb_lf,pe_ttm", "2005-01-03", "2018-04-16")
df_pb = getWsd(wsd)

wset = w.wset("sectorconstituent","date=2018-04-18;windcode=000852.SH")
df_code = getWset(wset)

codeList = df_code.wind_code.tolist()

l_rawdata = []

l_rawdata_c = l_rawdata.copy()

for code in codeList:
    wsd = w.wsd(code, "close,total_shares,profit_ttm2,tot_assets,tot_liab", "2014-06-30", "2018-04-17", "unit=1;rptType=1")
    df = getWsd(wsd)
    l_rawdata.append(df)


########################################################
########################################################
########################################################
df_raw = pd.read_csv("C:/Users/s_zhangyw/Desktop/raw.csv", index_col=0)    
df_fs = pd.read_csv("C:/Users/s_zhangyw/Desktop/fs_date.csv", index_col=0, encoding='GBK')



        

 

for gp in df_raw.groupby(by="code"):
    code = gp[0]
    data = gp[1]
    fs_date = df_fs.loc[code, :]
    for index, value in fs_date.iteritems():
        
    
    
   
l_pe = []
l_pb = []   
for gp in df_raw.groupby(by="code"):
    
    data = gp[1]
    pe = (data.CLOSE * data.TOTAL_SHARES).sum() / (data.PROFIT_TTM2).sum()
    pb = (data.CLOSE * data.TOTAL_SHARES).sum() / (data.TOT_ASSETS - data.TOT_LIAB).sum()
    l_pe.append(pe)
    l_pb.append(pb)
df_pbe = pd.DataFrame([l_pb, l_pe]).T
df_pbe.index = df.index

df_1 = df_pb.pct_change(1)
df_2 = df_pbe.pct_change(2)
df_2.columns = df_1.columns
x = df_2 - df_1 
