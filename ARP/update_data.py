# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:24:40 2018

@author: zhangyw49
"""
import numpy as np
import pandas as pd
import tdayfuncs as tdf
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
    data.shape
    
    index = out.Times
    col_level_1 = codeList
    col_level_2 = out.Fields * len(codeList)
    df = pd.DataFrame(data, index=index, columns=[col_level_1, col_level_2])
    return df

def update_to_csv(code_dict):
    for file_name, code_list in code_dict.items():
        df = pd.read_csv('./data/%s.csv'%file_name, index_col=[0], header=[0, 1], parse_dates=[0])
        start_date = tdf.tday_shift(df.index[-1].strftime("%Y-%m-%d"), -1)
        end_date = tdf.tday_shift(tdf.get_today(), -1)
        token = file_name.split(sep="_")[-1]
        if token == "ytm":
            df_update = get_edb(code_list, start_date, end_date)
        elif token == "pe":
            df_update = get_multi_wsd(code_list, "pe_ttm,ps_ttm,pb_lf", start_date, end_date)
        else:
            df_update = get_multi_wsd(code_list, "close", start_date, end_date)
        df_update.iloc[2:, :].to_csv('./data/%s.csv'%file_name, header=False, mode='a')
    return 0

def update_now(code_dict):
    w.start()
    try:
        update_to_csv(code_dict)
        print("\n************\nUpdate Done!\n************")
    except Exception as E:
        print(E)
    finally:
        w.stop()
    return 0
