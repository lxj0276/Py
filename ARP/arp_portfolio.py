# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:23:15 2018

@author: s_zhangyw
"""
import numpy as np
import pandas as pd
import tdayfuncs as tdf
from betas import weights_solver


strategy_id = [['stock_size_mom', 'stock_size_vol'], 
['stock_sector_mom', 'stock_sector_vol'], 
['bond_treasury_csm', 'bond_treasury_tsm', 'bond_treasury_value', 'bond_treasury_carry'],
['bond_finance_csm', 'bond_finance_tsm', 'bond_finance_value', 'bond_finance_carry'], 
['bond_corporate_csm', 'bond_corporate_tsm', 'bond_corporate_value', 'bond_corporate_carry'],
['multi_asset_rev', 'multi_asset_vol', 'multi_asset_value']]




def file_to_frame(file):
    df = pd.read_csv(file, index_col=[0], header=[0, 1], parse_dates=[0])
    return df


def get_order_days():
    trade_days = pd.to_datetime(tdf.get_trade_days()).to_series(name="Date")
    start, end = "2008-12-31", tdf.tday_shift(tdf.get_today(), -1)
    trade_days = trade_days[start:end]
    order_days = trade_days[::5]
    return order_days


def price_bind(strategy_id):
    df_price = pd.DataFrame()
    path = "./out/single/price/"
    unique_id = [s[0] for s in strategy_id]
    for u in unique_id:
        df = file_to_frame(path + u + '.csv')
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
        return df_wgt

    # 等权重模型
    pos_ew = df_rtn['2010':].copy()
    pos_ew.iloc[:, :] = 1 / pos_ew.shape[1]

    # 其他模型
    ## 最小波动率
    pos_mv = solver('min_vol', l_Sigma)
    ## 波动率平价
    pos_emv = solver('vol_parity', l_Sigma)
    ## 风险平价
    pos_rp = solver('risk_parity', l_Sigma)
    ## 风险预算  
    pos_rb = solver('risk_budget', l_r, l_Sigma)
    ## 最大分散化
    pos_md = solver('most_diversified', l_Sigma)
    ## 最大去相关性
    pos_decorr = solver('most_decorr', l_Rho)
    ## 最大夏普比
    pos_msr = solver('max_sharpe', l_r, l_Sigma)
    ## 均值-方差优化
    pos_mvo = solver('target_variance', l_r, l_Sigma)
    ## ReSample
    pos_smp, value_smp = solver('mv_resample', l_r, l_Sigma)
    #pos_smp, value_smp = pd.read_pickle('pos_smp.pkl'), pd.read_pickle('value_smp.pkl')

    pos_list = [pos_ew, pos_mv, pos_emv, pos_rp, pos_rb, pos_md, pos_decorr, pos_msr, pos_mvo, pos_smp]
    name_list = ['EW', 'MV', 'EMV', 'RP', 'RB', 'MD', 'DeCorr', 'MSR', 'MVO', 'ReSmp']
    return name_list, pos_list



df_rtn = return_bind(strategy_id)
order_day_list = get_order_days()["2010":].index.tolist()
pos_list, name_list = compute_portfolio(df_rtn, order_day_list)


def position_bind(strategy_id, df_weight):
    df_position = pd.DataFrame()
    path = "./out/single/position/"
    s_list = [s for strategy in strategy_id for s in strategy]
    for s in s_list:
        df = file_to_frame(path + s +'.csv')
        df = df.mul(df_weight.loc[:, s], axis=0)
        df = df.fillna(method="ffill").dropna()
        df_position = df.add(df_position, axis=0, fill_value=0)
    return df_position



########################################################################
########################################################################
########################################################################
########################################################################
########################################################################



plt.figure(dpi=300)
for value in value_list:
    plt.plot(value)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(name_list, fontsize=12, bbox_to_anchor=(1, 0.5), loc=6)
## 评价指标
for pos in pos_list:
    E = evaluation(pos.dropna(), df_rtn['2010':])
    print(E.YearRet())
for pos in pos_list:
    E = evaluation(pos.dropna(), df_rtn['2010':])
    print('####', E.CAGR(), E.Ret_Roll_1Y()[0], E.Ret_Roll_3Y()[0], E.Stdev(), E.Skew(), E.MaxDD(), E.MaxDD_Dur(), 
          E.VaR(), E.SR(), E.Calmar(), E.RoVaR(), E.Hit_Rate(), E.Gain2Pain(), sep='\n')

for pos in pos_list:
    E = evaluation(pos.dropna(), df_rtn['2010':])
    print(E.VaR())


#面积图、箱线图
plt.ioff() # plt.ion()
for pos in pos_list:
    plt.stackplot(pos.index, pos.values.T)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(pos.columns, fontsize=8, bbox_to_anchor=(1, 0.5), loc=6)
    plt.show()
    
for pos in pos_list:
    pos.plot.box(rot=45, ylim=(-0.1, 1.2), fontsize=12, showfliers=False)

#风险贡献
plt.ioff()
for pos in pos_list:
    df_rc = risk_contribution(pos.dropna())
    plt.stackplot(df_rc.index, df_rc.values.T)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(pos.columns, fontsize=8, bbox_to_anchor=(1, 0.5), loc=6)
    plt.show()
    
for pos in pos_list:
    df_rc = risk_contribution(pos.dropna())
    df_rc.plot.box(rot=45, ylim=(-0.1, 1.2), fontsize=12, showfliers=False)
 
