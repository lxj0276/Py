# 指数代码
# 策略类型
## 股票 mom vol
## 债券 csm tsm carry value
## 跨类 rev vol value


## 跨类指数：中证全指  恒生中国企业指数  中债-国债总财富(总值)指数 中债-金融债券总财富(总值)指数 中债-企业债AAA财富(总值)指数
asset_class_index = ['000985.CSI', 'HSCEI.HI', 'CBA00601.CS', 'CBA01201.CS', 'CBA04201.CS']


## 跨类 rev vol value
# 跨类rev算法
## 计算逻辑：计算对数收益率，某段历史正收益之和与负收益之和的比值，取负，取对数，再取负
## 简化之后，负收益之和与正收益之和的比值，取绝对值，取对数
def asset_rev(df, l=[240, 300, 360]):
    log_rtn = np.log(df).diff(1)
    def rev(lag):
        return log_rtn.rolling(lag).apply(lambda x: np.log(-np.sum(x[x<0]) / np.sum(x[x>0])), raw=True)
    return rev(l[0]) + rev(l[1]) + rev(l[2])

# 跨类vol算法
## 计算逻辑：对数收益率，三段标准差的均值，三段区间标准差的变化率，再取均值
def asset_vol(df, l=[60, 120, 180]):
    log_rtn = np.log(df).diff(1)
    def std(lag):
        return log_rtn.rolling(lag).std()
    std = std(l[0]) + std(l[1]) + std(l[2])
    def vol(lag):
        return std.diff(lag)
    return vol(l[0]) + vol(l[1]) + vol(l[2])


# 跨类value算法
## 计算逻辑：对于两个股票指数，100 / pe, pb, ps, 最近五日平均，历史所有score, 再取平均，对于债券指数，YTM五日平均，取历史所有score##
##准备数据， pe, pb, ps, 和 ytm
## 一步一步来
## 
## 股票估值
## 债券估值

def asset_value_zscore(df, m=5):
    roll = df.rolling(5).mean()
    mean = roll.expanding().mean()
    std = roll.expanding().std()
    zscore = (roll - mean) / std
    return zscore

def asset_value(df, df_value_stock, df_value_bond):
    df_stock = asset_value_zscore(df_value_stock)
    df_stock = df_stock.mean(axis=1, level=0)
    df_bond = asset_value_zscore(df_value_bond)
    df_value = pd.concat([df_stock, df_bond], axis=1)
    df_value.columns = df.columns
    return df_value



asset_value_stock = pd.read_csv('asset_value_stock.csv', index_col=[0], header=[0, 1], parse_dates=[0])
asset_value_bond = pd.read_csv('asset_value_bond.csv', index_col=[0], header=[0], parse_dates=[0])



# 交易日期文件
trade_days_file = "C:/Users/zhangyw49/code/py/tDays.csv"
df_date = pd.read_csv(trade_days_file, parse_dates=[0], index_col=[0], header=[0])
order_date = df_date["20081231":"20180531"].iloc[::5, :]
# 由order_date控制交易信号

# 计算溢价策略表现，因子值要统一为越大越好
def net_value(df_price, df_factor, position=[0.5, 0.5, -0.5, -0.5]):
    price, factor = df_price["20081231":], df_factor["20081231":]
    # T-1日末发出信号，T日以收盘价建仓
    df_rank = factor.rank(axis=1, ascending=False, na_option="top")
    df_position = df_rank.applymap(lambda x: position[int(x)-1])
    # 从T+1日起收益开始计算
    df_rtn = price.pct_change(1).fillna(0)
    df_pos = df_position.loc[order_date.index, :]
    df_pos = df_pos.reindex(df_rtn.index).fillna(method="bfill")
    sr_rtn = (df_pos.shift(1) * df_rtn).sum(axis=1)
    return sr_rtn


asset_class = pd.read_csv('asset_class.csv', index_col=[0], header=[0, 1], parse_dates=[0])

asset_calss_rev = asset_rev(asset_class)
asser_class_vol = asset_vol(asset_class)

asser_class_value = asset_value(asset_class, asset_value_stock, asset_value_bond)




