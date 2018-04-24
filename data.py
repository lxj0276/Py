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
for code in codeList[:3]:
    wsd = w.wsd(code, "close,total_shares,profit_ttm2,tot_assets,tot_liab", "2014-01-01", "2018-04-17", "unit=1;rptType=1")
    df = get_wsd(wsd)
    df['code'] = code
    df_raw = df_raw.append(df)

df_raw.to_csv('raw.csv')
pd.to_pickle(df_raw, 'raw.pkl')


###################

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

wset = w.wset("sectorconstituent","date=2014-10-17;sectorid=1000012163000000")
df_code = get_wset(wset)

codeList = df_code.wind_code.tolist()

df_raw = pd.DataFrame()
for code in codeList:
    wsd = w.wsd(code, "mkt_cap_ard,netprofit_ttm,equity_new,or_ttm,operatecashflow_ttm", "2005-01-03", "2014-12-31", "unit=1")
    df = get_wsd(wsd)
    df['code'] = code
    df_raw = df_raw.append(df)
    print(code)

df_raw.to_csv('raw.csv')
df_raw.to_pickle('raw.pkl')

w.stop()

##############################################
df_raw = pd.read_pickle('raw.pkl')
df_raw.head()
sr_pe = df_raw.iloc[:]

df = df_raw.reset_index()

df_sum = pd.DataFrame()
for gp in df.groupby("index"):
    date = gp[0]
    data = gp[1]
    sr = data.iloc[:, 1:-1].sum()
    sr.name=date
    df_sum = pd.concat([df_sum, sr], axis=1)
df_sum = df_sum.T

df_value = df_sum.apply(lambda x: x['MKT_CAP_ARD'] / x, axis=1).drop('MKT_CAP_ARD', axis=1)
df_value.columns = ["PB", "PE", "PCF", "PS"]
