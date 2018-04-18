# -*- coding: utf-8 -*-

# get data from Wind
import numpy as np
import pandas as pd
from WindPy import w
from datetime import *
w.start()

def get_wsd(wsd):
    df = pd.DataFrame(wsd.Data).T
    df.index = wsd.Times
    df.columns = wsd.Fields 
    return df

def get_wset(wset):
    df = pd.DataFrame(wset.Data).T
    df.columns = wset.Fields
    df.set_index('date', inplace=True)
    return df

wsd = w.wsd("000852.SH", "pb_lf,pe_ttm", "2005-01-03", "2018-04-16")
df_pbpe = get_wsd(wsd)

wset = w.wset("sectorconstituent","date=2018-04-18;windcode=000852.SH")
df_code = get_wset(wset)

codeList = df_code.wind_code.tolist()

df_raw = pd.DataFrame()
for code in codeList:
    wsd = w.wsd(code, "close,total_shares,profit_ttm2,tot_assets,tot_liab", "2014-01-01", "2018-04-17", "unit=1;rptType=1")
    df = get_wsd(wsd)
    df['code'] = code
    df_raw = df_raw.append(df)

df_raw.to_csv('raw.csv')
pd.to_pickle(df_raw, 'raw.pkl')