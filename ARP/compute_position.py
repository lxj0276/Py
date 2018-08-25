# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:16:00 2018

@author: zhangyw49
"""

import numpy as np
import pandas as pd
from parameters import param_dict

# 因子值从大到小排列时，与positions顺序对应
def factor_to_position(df_factor, positions):
    df_factor = df_factor.shift(1).dropna()
    df_rank = df_factor.rank(axis=1, method='first', ascending=False)
    df_position = df_rank.applymap(lambda x: positions[int(x)-1])
    return df_position
    

# 股票mom计算方法
## 计算逻辑：最近五日均值与历史L天最大值之比，最后再取平均
def stock_mom_factor(df, m, lags):
    def mom(lag):
        return df.rolling(m).mean() / df.rolling(lag, min_periods=10).max().shift(1) - 1
    df_factor = mom(lags[0]) + mom(lags[1]) + mom(lags[2])
    return df_factor

def stock_mom(df, *, m=5, lags=[210, 240, 270], positions=[0.5, 0.5, -0.5, -0.5]):
    df_factor = stock_mom_factor(df, m, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position

# 股票vol计算方法
## 计算逻辑：最近五日波动率均值与L天之前的五日均值之比，最后取平均
def stock_vol_factor(df, m, lags):
    log_rtn = np.log(df).diff(1)
    std = log_rtn.expanding().std()
    def vol(lag):
        return -std.rolling(m).mean() / std.rolling(m).mean().shift(lag)
    df_factor = vol(lags[0]) + vol(lags[1]) + vol(lags[2])
    return df_factor

def stock_vol(df, *, m=5, lags=[240, 270, 300], positions=[0.5, 0.5, -0.5, -0.5]):
    df_factor = stock_vol_factor(df, m, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


# 债券csm计算方法
## 计算逻辑：L天的收益率，取平均
def bond_csm_factor(df, lags):
    def csm(lag):
        return df.pct_change(lag)
    df_factor = (csm(lags[0]) + csm(lags[1]) + csm(lags[2])).shift(1)
    return df_factor

def bond_csm(df, *, lags=[20, 40, 60], positions=[0.5, 0.5, 0, -0.5, -0.5]):
    df_factor = bond_csm_factor(df, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


# 债券tsm计算方法
## 计算逻辑：构造DR007指数，计算三个周期的ts动量。根据动量，替换长端，买长卖短。
def bond_tsm_rf(df_rf_rate):
    delta_day = (df_rf_rate.index[1:] - df_rf_rate.index[:-1]).days.tolist()
    df_delta_day = pd.DataFrame(np.array(delta_day + [0]), index=df_rf_rate.index, columns=df_rf_rate.columns)
    df_rf = (1 + df_rf_rate / (365 * 100) * df_delta_day).cumprod()
    return df_rf

def bond_tsm_factor(df, df_rf, lags):
    df_rf = bond_tsm_rf(df_rf)
    df_bind = pd.concat([df, df_rf], axis=1)
    df_bind = df_bind.fillna(method="ffill")
    def tsm(lag):
        return df_bind.pct_change(lag)
    df_tsm = (tsm(lags[0]) + tsm(lags[1]) + tsm(lags[2])).shift(1)
    return df_bind.dropna(), df_tsm.dropna()

def bond_tsm_position(df_tsm, positions):
    df = df_tsm.shift(1).copy()
    df.iloc[:, :3] = 1
    df_long = df.iloc[:, -3:]
    df_long_rf = df_long.subtract(df_long.iloc[:, -1], axis=0)
    df.iloc[:, -3:] = df_long_rf.applymap(lambda x: 1 if x > 0 else 0)
    n = len(positions)
    df.iloc[:, -1] = n - df.sum(axis=1)
    df_position = df.multiply(np.array(positions+[positions[-1]]))
    return df_position

def bond_tsm(df, df_rf, *, lags=[20, 40, 60], positions=[-0.5, -0.5, 0, 0.5, 0.5]):
    df_price, df_factor = bond_tsm_factor(df, df_rf, lags)
    df_position = bond_tsm_position(df_factor, positions)
    return df_price, df_position

# 债券value计算方法
## 计算逻辑：YTM五日平均，分别取50，100，150日最大值的比率平均
def bond_value_ytm(df, ytm_origin, transform_matrix):
    transform_matrix = np.array(transform_matrix).T
    ytm_transform = ytm_origin.dot(transform_matrix)
    ytm_transform.columns = df.columns
    return ytm_transform 

def bond_value_factor(ytm_transform, m, lags):
    df_avg = ytm_transform.rolling(m).mean()
    def value(lag):
        df = df_avg / df_avg.rolling(lag).max()
        return df
    df_value = value(lags[0]) + value(lags[1]) + value(lags[2])
    return df_value

def bond_value(df, ytm_origin, *, transform_matrix, m=5, lags=[50, 100, 150], positions=[0.5, 0.5, 0, -0.5, -0.5]):
    ytm_transform = bond_value_ytm(df, ytm_origin, transform_matrix)
    df_factor = bond_value_factor(ytm_transform, m, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


# 债券carry计算方法
# 1.首先，五个国债指数及其收益
# 2.需要数据，2，4，6，8，9，10 20 以及DR007 的YTM
# 3.YTM滚动10日平均
# 4. YTM对对应期限的斜率
# 5. 对斜率， 100 * （最近60天 - 之前750天），以及500天和250天
# 6. 对上述求均值。并称之为套索
# 7. 套索为正，买短卖长，套索为负，买长卖短
def bond_carry_ytm(df, df_rf, ytm_origin, transform_matrix):
    transform_matrix = np.array(transform_matrix).T
    ytm_transform = ytm_origin.dot(transform_matrix)
    ytm_transform.columns = df.columns
    ytm_transform = pd.concat([df_rf, ytm_transform], axis=1)
    ytm_transform.fillna(method="ffill", inplace=True)
    return ytm_transform.dropna()

def bond_carry_facotr(ytm_transform, m, x, lags):
    def slope(x, y):
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        slope = np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered)
        return slope
    df_slope = ytm_transform.apply(lambda y: slope(x, y), axis=1)
    def carry(lag):
        df = (df_slope.rolling(m).mean() - df_slope.shift(lag).rolling(m).mean()) * 100
        return df
    df_carry = carry(lags[0]) + carry(lags[1]) + carry(lags[2]) 
    return df_carry

def bond_carry_position(df, df_carry, positions):
    df_ones = pd.DataFrame(np.ones((df.shape[0], df.shape[1])), index=df.index, columns=df.columns)
    df_position = df_ones * positions
    df_factor = df_carry.shift(1)[df.index].apply(lambda x: -1 if x > 0 else 1)
    df_position = df_position.multiply(df_factor, axis=0)
    return df_position

def bond_carry(df, df_rf, ytm_origin, *, transform_matrix, x=[7/365, 2, 4, 6, 17/2, 50/3], m=60, lags=[250, 500, 750], positions=[0.5, 0.5, 0, -0.5, -0.5]):
    ytm_transform = bond_carry_ytm(df, df_rf, ytm_origin, transform_matrix)
    df_carry = bond_carry_facotr(ytm_transform, m, x, lags)
    df_position = bond_carry_position(df, df_carry, positions)
    df_price = df.copy()
    return df_price, df_position


# 跨类rev计算方法
## 计算逻辑：计算对数收益率，某段历史正收益之和与负收益之和的比值，取负，取对数，再取负
## 简化之后，负收益之和与正收益之和的比值，取绝对值，取对数
def multi_rev_facotr(df, lags):
    log_rtn = np.log(df).diff(1)
    def rev(lag):
        return log_rtn.rolling(lag).apply(lambda x: np.log(-np.sum(x[x<0]) / np.sum(x[x>0])), raw=True)
    df_factor = rev(lags[0]) + rev(lags[1]) + rev(lags[2])
    return df_factor

def multi_rev(df, *, lags=[240, 300, 360], positions=[0.5, 0,5, 0, -0.5, -0.5]):
    df_factor = multi_rev_facotr(df, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


# 跨类vol计算方法
## 计算逻辑：对数收益率，三段标准差的均值，三段区间标准差的变化率，再取均值
def multi_vol_factor(df, lags=[60, 120, 180]):
    log_rtn = np.log(df).diff(1)
    def std(lag):
        return log_rtn.rolling(lag).std()
    std = std(lags[0]) + std(lags[1]) + std(lags[2])
    def vol(lag):
        return std.diff(lag)
    df_factor = -1 * (vol(lags[0]) + vol(lags[1]) + vol(lags[2]))
    return df_factor

def multi_vol(df, *, lags=[60, 120, 180], positions=[0.5, 0,5, 0, -0.5, -0.5]):
    df_factor = multi_vol_factor(df, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


# 跨类value计算方法
## 计算逻辑：对于两个股票指数，100 / pe, pb, ps, 最近五日平均，历史所有score, 再取平均，对于债券指数，YTM五日平均，取历史所有score##

def multi_value_zscore(df, m):
    roll = df.rolling(m).mean()
    mean = roll.expanding().mean()
    std = roll.expanding().std()
    zscore = (roll - mean) / std
    return zscore

def multi_value_factor(df, df_value_stock, df_value_bond, m):
    df_stock = multi_value_zscore(100 / df_value_stock, m)
    df_stock = df_stock.mean(axis=1, level=0)
    df_bond = multi_value_zscore(df_value_bond, m)
    df_factor = pd.concat([df_stock, df_bond], axis=1)
    df_factor.columns = df.columns
    return df_factor

def multi_value(df, df_value_stock, df_value_bond, *, m=5, positions=[0.5, 0,5, 0, -0.5, -0.5]):
    df_factor = multi_value_factor(df, df_value_stock, df_value_bond, m)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position



def file_to_frame(file_name):
    df = pd.read_csv('./data/%s.csv'%file_name, index_col=[0], header=[0, 1], parse_dates=[0])
    return df

def save_file(price, position, file_name):
    price.to_csv('./out/single/price/%s.csv'%file_name)
    position.to_csv('./out/single/position/%s.csv'%file_name)
    print("\n%s: Computing..."%file_name)
    return None


def compute_position(param_dict=param_dict):
    stock_size = file_to_frame("stock_size")
    stock_sector = file_to_frame("stock_sector")
    bond_treasury = file_to_frame("bond_treasury")
    bond_finance = file_to_frame("bond_finance")
    bond_corporate = file_to_frame("bond_corporate")
    multi_asset = file_to_frame("multi_asset")
    bond_dr007_rate = file_to_frame("bond_dr007_rate")
    bond_treasury_ytm = file_to_frame("bond_treasury_ytm")
    bond_finance_ytm = file_to_frame("bond_finance_ytm")
    bond_corporate_ytm = file_to_frame("bond_corporate_ytm")
    multi_asset_ytm = file_to_frame("multi_asset_ytm")
    multi_asset_pe = file_to_frame("multi_asset_pe")


    price, position = stock_mom(stock_size, **param_dict["stock"]["size"]["mom"])
    save_file(price, position, "stock_size_mom")
    
    price, position = stock_vol(stock_size, **param_dict["stock"]["size"]["vol"])
    save_file(price, position, "stock_size_vol")
    
    price, position = stock_mom(stock_sector, **param_dict["stock"]["sector"]["mom"])
    save_file(price, position, "stock_sector_mom")
    
    price, position = stock_vol(stock_sector, **param_dict["stock"]["sector"]["vol"])
    save_file(price, position, "stock_sector_vol")


    price, position = bond_csm(bond_treasury, **param_dict["bond"]["treasury"]["csm"])
    save_file(price, position, "bond_treasury_csm")

    price, position = bond_tsm(bond_treasury, bond_dr007_rate, **param_dict["bond"]["treasury"]["tsm"])
    save_file(price, position, "bond_treasury_tsm")
    
    price, position = bond_value(bond_treasury, bond_treasury_ytm, **param_dict["bond"]["treasury"]["value"])
    save_file(price, position, "bond_treasury_value")

    price, position = bond_carry(bond_treasury, bond_dr007_rate, bond_treasury_ytm, **param_dict["bond"]["treasury"]["carry"])
    save_file(price, position, "bond_treasury_carry")

    price, position = bond_csm(bond_finance, **param_dict["bond"]["finance"]["csm"])
    save_file(price, position, "bond_finance_csm")

    price, position = bond_tsm(bond_finance, bond_dr007_rate, **param_dict["bond"]["finance"]["tsm"])
    save_file(price, position, "bond_finance_tsm")

    price, position = bond_value(bond_finance, bond_finance_ytm, **param_dict["bond"]["finance"]["value"])
    save_file(price, position, "bond_finance_value")

    price, position = bond_carry(bond_finance, bond_dr007_rate, bond_finance_ytm, **param_dict["bond"]["finance"]["carry"])
    save_file(price, position, "bond_finance_carry")
    
    price, position = bond_csm(bond_corporate, **param_dict["bond"]["corporate"]["csm"])
    save_file(price, position, "bond_corporate_csm")

    price, position = bond_tsm(bond_corporate, bond_dr007_rate, **param_dict["bond"]["corporate"]["tsm"])
    save_file(price, position, "bond_corporate_tsm")

    price, position = bond_value(bond_corporate, bond_corporate_ytm, **param_dict["bond"]["corporate"]["value"])
    save_file(price, position, "bond_corporate_value")

    price, position = bond_carry(bond_corporate, bond_dr007_rate, bond_corporate_ytm, **param_dict["bond"]["corporate"]["carry"])
    save_file(price, position, "bond_corporate_carry")
    

    price, position = multi_rev(multi_asset, **param_dict["multi"]["asset"]["rev"])
    save_file(price, position, "multi_asset_rev")
    
    price, position = multi_vol(multi_asset, **param_dict["multi"]["asset"]["vol"])
    save_file(price, position, "multi_asset_vol")
    
    price, position = multi_value(multi_asset, multi_asset_pe, multi_asset_ytm, **param_dict["multi"]["asset"]["value"])
    save_file(price, position, "multi_asset_value")
    

    print("\nComputation Done!")



if __name__ =="__main__":
    compute_position(param_dict)