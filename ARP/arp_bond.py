# 指数代码
# 策略类型
## 股票 mom vol
## 债券 csm tsm carry value
## 跨类 rev vol value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


bond_treasury_index = ['CBA00621.CS', 'CBA00631.CS', 'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS']
## 金融债财富指数：中债-金融债券总财富(1-3年)指数   中债-金融债券总财富(3-5年)指数  中债-金融债券总财富(5-7年)指数  中债-金融债券总财富(7-10年)指数 中债-金融债券总财富(10年以上)指数
bond_finance_index = ['CBA01221.CS', 'CBA01231.CS', 'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS']
## AAA企业债财富指数：中债-企业债AAA财富(1-3年)指数   中债-企业债AAA财富(3-5年)指数 中债-企业债AAA财富(5-7年)指数 中债-企业债AAA财富(7-10年)指数    中债-企业债AAA财富(10年以上)指数
bond_corporate_index = ['CBA04221.CS', 'CBA04231.CS', 'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS']



# 交易日期文件
trade_days_file = "C:/Users/zhangyw49/code/py/tDays.csv"
df_date = pd.read_csv(trade_days_file, parse_dates=[0], index_col=[0], header=[0])
order_date = df_date["20081231":"20180531"].iloc[::5, :]
# 由order_date控制交易信号


# 债券csm算法
# 计算逻辑：L天的收益率，取平均
# 需要数据：价格
def bond_csm(df, l=[20, 40, 60]):
    def csm(lag):
        return df.pct_change(lag)
    return (csm(l[0]) + csm(l[1]) + csm(l[2])).shift(1)


# 债券tsm算法
# 计算逻辑：构造DR007指数，计算三个周期的ts动量
# 所需数据：无风险利率
def bond_tsm_factor(df, df_rf, l):
    df_rf.columns = ["rf"]
    df_bind = pd.concat([df, df_rf], axis=1)
    def tsm(lag):
        return df_bind.pct_change(lag)
    df_tsm = (tsm(l[0]) + tsm(l[1]) + tsm(l[2])).shift(1)
    return df_bind, df_tsm

def bond_tsm_pos(df_tsm):
    df = df_tsm.subtract(df_tsm.rf, axis=0)
    def pos(x):
        if x<= 0:
            return 0
        elif x > 0:
            return 1
        else:
            return np.nan
    df = df.applymap(lambda x: pos(x))
    n = df.shape[1] - 1
    df["rf"] = n - df.sum(axis=1)
    df = df / n
    return df

def bond_tsm(df, df_rf, l=[20, 40, 60]):
    df_bind, df_tsm = bond_tsm_factor(df, df_rf, l)
    df_pos = bond_tsm_pos(df_tsm)
    return df_bind, df_pos

# 债券value算法
# 计算逻辑：YTM五日平均，分别取50，100，150日最大值的比率平均
def bond_value_ytm(df, ytm_origin, T_matrix):
    ytm_transform = ytm_origin.dot(T_matrix)
    ytm_transform.columns = df.columns
    return ytm_transform 

def bond_value_factor(ytm_transform, m, l):
    df_avg = ytm_transform.rolling(m).mean()
    def value(lag):
        df = df_avg / df_avg.rolling(lag).max()
        return df
    df_value = value(l[0]) + value(l[1]) + value(l[2])
    return df_value

def bond_value(df, df_ytm, T_matrix, m=5, l=[50, 100, 150]):
    ytm_transform = bond_value_ytm(df, df_ytm, T_matrix)
    df_value = bond_value_factor(ytm_transform, m ,l)
    return df_value


# 债券carry算法
# 一步一步来
# 1.首先，五个国债指数及其收益
# 2.需要数据，2，4，6，8，9，10 20 以及DR007 的YTM
# 3.YTM滚动10日平均
# 4. YTM对对应期限的斜率
# 5. 对斜率， 100 * （最近60天 - 之前750天），以及500天和250天
# 6. 对上述求均值。并称之为套索
# 7. 套索为正，买长，套索为负，买短
def bond_carry_ytm(df, df_rf, ytm_origin, T_matrix):
    ytm_transform = ytm_origin.dot(T_matrix)
    ytm_transform.columns = df.columns
    df_rf.columns = ['rf']
    ytm_transform = pd.concat([df_rf, ytm_transform], axis=1)
    ytm_transform.fillna(method="ffill", inplace=True)
    return ytm_transform

def bond_carry_facotr(ytm_transform, m, x, l):
    def slope(x, y):
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        slope = np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered)
        return slope
    df_slope = ytm_transform.apply(lambda y: slope(x, y), axis=1)
    
    def carry(lag):
        df = (df_slope.rolling(m).mean() - df_slope.shift(lag).rolling(m).mean()) * 100
        return df
    df_carry = carry(l[0]) + carry(l[1]) + carry(l[2])
    
    return df_carry

def bond_carry_pos(df, df_carry, pos):
    df_ones = pd.DataFrame(np.ones((df.shape[0], df.shape[1])), index=df.index, columns=df.columns)
    df_pos = df_ones * pos
    df_pos = np.where(df_carry>0, df_pos, -1 * df_pos)
    return df_pos

def bond_carry(df, df_rf, ytm_origin, T_matrix, pos, m=60, x=[7/365, 2, 4, 6, 17/2, 50/3], l=[250, 500, 750]):
    ytm_transform = bond_carry_ytm(df, df_rf, ytm_origin, T_matrix)
    df_carry = bond_carry_facotr(ytm_transform, m, x, l)
    df_pos = bond_carry_pos(df, df_carry, pos)
    return df, df_pos

df = bond_corporate
df_rf = bond_dr007
ytm_origin = bond_corporate_ytm
T_matrix=T_matrix_corporate
pos = position_corporate

bond_corporate_carry_price,  bond_corporate_carry_pos = bond_carry(, , bond_corporate_ytm, T_matrix_corporate, position_corporate)


# 准备工作
# 首先读取DR007 和 YTM,# 构造DR007指数
bond_dr007_rate = pd.read_csv('bond_dr007_rate.csv', index_col=[0], header=[0], parse_dates=[0])
bond_dr007 = (1 + bond_dr007_rate / (100 * 360)).cumprod()

bond_treasury_ytm = pd.read_csv('bond_treasury_ytm.csv', index_col=[0], header=[0], parse_dates=[0])
bond_finance_ytm = pd.read_csv('bond_finance_ytm.csv', index_col=[0], header=[0], parse_dates=[0])
bond_corporate_ytm = pd.read_csv('bond_corporate_ytm.csv', index_col=[0], header=[0], parse_dates=[0])


########################################
bond_treasury = pd.read_csv('bond_treasury.csv', index_col=[0], header=[0, 1], parse_dates=[0])
## 计算净值，分为两种，一种给因子，一种直接给仓位
# 1
bond_treasury_csm = bond_csm(bond_treasury)
# 2
bond_treasury_tsm_price,  bond_treasury_tsm_pos = bond_tsm(bond_treasury, bond_dr007)
# 3
T_matrix_treasury = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]]).T
bond_treasury_value = bond_value(bond_treasury, bond_treasury_ytm, T_matrix_treasury)
# 4
position_treasury = [-0.5, -0.5, 0, 0.5, 0.5]
bond_treasury_carry_price,  bond_treasury_carry_pos = bond_carry(bond_treasury, bond_dr007, bond_treasury_ytm, T_matrix_treasury, position_treasury)



##############################
bond_finance = pd.read_csv('bond_finance.csv', index_col=[0], header=[0, 1], parse_dates=[0])
## 计算净值，分为两种，一种给因子，一种直接给仓位
# 1
bond_finance_csm = bond_csm(bond_finance)
# 2
bond_finance_tsm_price,  bond_finance_tsm_pos = bond_tsm(bond_finance, bond_dr007)
# 3
T_matrix_finance = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/3,2/3]]).T
bond_finance_value = bond_value(bond_finance, bond_finance_ytm, T_matrix_finance)
# 4
position_finance = [-0.5, -0.5, 0, 0.5, 0.5]
bond_finance_carry_price,  bond_finance_carry_pos = bond_carry(bond_finance, bond_dr007, bond_finance_ytm, T_matrix_finance, position_finance)




##############################
bond_corporate = pd.read_csv('bond_corporate.csv', index_col=[0], header=[0, 1], parse_dates=[0])
## 计算净值，分为两种，一种给因子，一种直接给仓位
# 1
bond_corporate_csm = bond_csm(bond_corporate)
# 2
bond_corporate_tsm_price,  bond_corporate_tsm_pos = bond_tsm(bond_corporate, bond_dr007)
# 3
T_matrix_corporate = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1/2,1/2,0,0], [0,0,0,0,0,1/2,1/2]]).T
bond_corporate_value = bond_value(bond_corporate, bond_corporate_ytm, T_matrix_corporate)
# 4
position_corporate = [-0.5, -0.5, 0, 0.5, 0.5]
bond_corporate_carry_price,  bond_corporate_carry_pos = bond_carry(bond_corporate, bond_dr007, bond_corporate_ytm, T_matrix_corporate, position_corporate)


