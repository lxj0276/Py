# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:23:15 2018

@author: s_zhangyw
"""
import numpy as np
import pandas as pd
import tdayfuncs as tdf
from betas import weights_solver
from parameters import trade_cost



def file_to_frame(file):
    df = pd.read_csv(file, index_col=[0], header=[0, 1], parse_dates=[0])
    return df

def get_order_days():
    trade_days = pd.to_datetime(tdf.get_trade_days()).to_series(name="Date")
    start, end = "2008-12-31", tdf.tday_shift(tdf.get_today(), -1)
    trade_days = trade_days[start:end]
    order_days = trade_days[::5]
    return order_days



strategy_id = [['stock_size_mom', 'stock_size_vol'], 
['stock_sector_mom', 'stock_sector_vol'], 
['bond_treasury_csm', 'bond_treasury_tsm', 'bond_treasury_value', 'bond_treasury_carry'],
['bond_finance_csm', 'bond_finance_tsm', 'bond_finance_value', 'bond_finance_carry'], 
['bond_corporate_csm', 'bond_corporate_tsm', 'bond_corporate_value', 'bond_corporate_carry'],
['multi_asset_rev', 'multi_asset_vol', 'multi_asset_value']]


def price_bind(strategy_id):
    df_price = pd.DataFrame()
    path_price = "./out/single/price/"
    unique_id = [s[0] for s in strategy_id]
    for u in unique_id:
        df = file_to_frame(path_price + u + '.csv')
        df_price = pd.concat([df_price, df], axis=1)
    df = file_to_frame(path + "bond_treasury_tsm" + '.csv')
    df_price = pd.concat([df_price, df.iloc[:, [-1]]], axis=1)
    return df_price

def return_bind(strategy_id):
    path_return = "./out/single/return/"
    s_list = [s for strategy in strategy_id for s in strategy]
    df_rtn = pd.DataFrame()
    for s in s_list:
        df = pd.read_csv(path_return+s+'.csv', index_col=0, parse_dates=[0])
        df_rtn = pd.concat([df_rtn, df], axis=1)
    df_rtn.columns = s_list
    return df_rtn

def position_bind(strategy_id, df_weight):
    path = "./out/single/position/"
    s_list = [s for strategy in strategy_id for s in strategy]
    df = file_to_frame(path + s_list[0] + ".csv")
    df_position = pd.DataFrame(index=df.index, columns=df.columns)
    df_weight.reindex(df_position.index).fillna(method="ffill")
    for s in s_list:
        df = file_to_frame(path + s +'.csv')
        df = df.mul(df_weight.loc[:, s], axis=0)
        df = df.fillna(method="ffill").dropna()
        df_position = df_position.add(df, axis=0, fill_value=0)
    return df_position


def position_to_return(price, position, trade_cost=None):
    rtn = price.pct_change(1)
    if not trade_cost:
        rtn_sum = (position.shift(1) * rtn).sum(axis=1)
    else:
        trade_cost = np.array(trade_cost)
        rtn_after = rtn - position.diff(1).shift(1).abs().mul(trade_cost/2, axis=1)
        rtn_sum = (position.shift(1) * rtn_after).sum(axis=1)
    return rtn_sum.dropna()


def compute_portfolio(df_rtn, order_day_list):
    # 收益率和协方差的预测
    lag = len(df_rtn["2009"])
    rtn_p = df_rtn.shift(1).rolling(lag).mean()
    rho_p = df_rtn.shift(1).ewm(lag).corr()
    cov_p = df_rtn.shift(1).ewm(lag).cov()

    # 转换为list
    l_day = order_day_list
    l_r = []
    l_Rho = []
    l_Sigma = []
    for d in l_day:
        r = np.array(rtn_p.loc[d])
        Rho = np.array(rho_p.loc[d])
        Sigma = np.array(cov_p.loc[d])
        l_r.append(r)
        l_Rho.append(Rho)
        l_Sigma.append(Sigma)

    # 仓位、净值
    def solver(func_name, *args):
        l_weights = weights_solver(func_name, *args)
        df_wgt = pd.DataFrame(l_weights, l_day, columns=df_rtn.columns)
        print("\n%s computing..."%func_name)
        return df_wgt

    # 等权重模型
    wgt_ew = df_rtn['2010':].copy()
    wgt_ew.iloc[:, :] = 1 / wgt_ew.shape[1]

    # 其他模型
    ## 最小波动率
    wgt_mv = solver('min_vol', l_Sigma)
    ## 波动率平价
    wgt_emv = solver('vol_parity', l_Sigma)
    ## 风险平价
    wgt_rp = solver('risk_parity', l_Sigma)
    ## 风险预算  
    wgt_rb = solver('risk_budget', l_r, l_Sigma)
    ## 最大分散化
    wgt_md = solver('most_diversified', l_Sigma)
    ## 最大去相关性
    wgt_decorr = solver('most_decorr', l_Rho)
    ## 最大夏普比
    wgt_msr = solver('max_sharpe', l_r, l_Sigma)
    ## 均值-方差优化
    wgt_mvo = solver('target_variance', l_r, l_Sigma)
    ## ReSample
    wgt_smp = solver('mv_resample', l_r, l_Sigma)
    #wgt_smp, value_smp = pd.read_pickle('wgt_smp.pkl'), pd.read_pickle('value_smp.pkl')

    weight_list = [wgt_ew, wgt_mv, wgt_emv, wgt_rp, wgt_rb, wgt_md, wgt_decorr, wgt_msr, wgt_mvo, wgt_smp]
    name_list = ['EW', 'MV', 'EMV', 'RP', 'RB', 'MD', 'DeCorr', 'MSR', 'MVO', 'ReSmp']
    return name_list, weight_list



def compute_now():
    df_rtn = return_bind(strategy_id)

    order_day_list = get_order_days()["2010":].index.tolist()
    name_list, weight_list = compute_portfolio(df_rtn, order_day_list)

    df_price = price_bind(strategy_id)
    df_price.to_csv('./out/portfolio/price/price_bind.csv')
    for name, weight in zip(name_list, weight_list):
        df_position = position_bind(strategy_id, weight)
        df_position.to_csv("./out/portfolio/position/%s.csv"%name)
        df_return = position_to_return(df_price, df_position, trade_cost)
        df_return.to_csv("./out/portfolio/return/%s.csv"%name)


    
if __name__ == '__main__':
    compute_now()