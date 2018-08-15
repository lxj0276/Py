# 指数代码
# 策略类型
## 股票 mom vol
## 债券 csm tsm carry value
## 跨类 rev vol value

## 股票规模指数：上证50 沪深300 中证500 中证1000
stock_size_index = ['000016.SH', '000300.SH', '000852.SH', '000905.SH']
## 股票行业申万指数：农林牧渔 采掘 化工 钢铁 有色金属 电子 家用电器 食品饮料 纺织服装 轻工制造 医药生物 公用事业 交通运输 房地产商业贸易 休闲服务 综合 建筑材料 建筑装饰 电气设备 国防军工 计算机传媒 通信 银行 非银金融 汽车 机械设备
stock_sector_index = ['801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI', '801080.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI', '801150.SI', '801160.SI', '801170.SI',
                      '801180.SI', '801200.SI', '801210.SI', '801230.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI', '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI', '801880.SI', '801890.SI']

# 股票mom算法
# 计算逻辑：最近五日均值与历史L天最大值之比，最后再取平均
def stock_mom(df, m=5, l=[210, 240, 270]):
    def mom(lag):
        return df.rolling(m).mean() / df.rolling(lag).max().shift(1) - 1
    
    return mom(l[0]) + mom(l[1]) + mom(l[2])


# 股票mom算法
# 计算逻辑：最近五日均值与历史L天最大值之比，最后再取平均
def stock_mom(df, m=5, l=[210, 240, 270]):
    def mom(lag):
        return df.rolling(m).mean() / df.rolling(lag, min_periods=10).max().shift(1) - 1
    return mom(l[0]) + mom(l[1]) + mom(l[2])

# 股票vol算法
# 计算逻辑：最近五日波动率均值与L天之前的五日均值之比，最后取平均
def stock_vol(df, m=5, l=[240, 270, 300]):
    log_rtn = np.log(df).diff(1)
    std = log_rtn.expanding().std()
    def vol(lag):
        return -std.rolling(m).mean() / std.rolling(m).mean().shift(lag)
    return vol(l[0]) + vol(l[1]) + vol(l[2])


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

## 计算因子
stock_size = pd.read_csv('stock_size.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_size_mom = stock_mom(stock_size)
stock_size_vol = stock_vol(stock_size)

stock_sector = pd.read_csv('stock_sector.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_sector_mom = stock_mom(stock_sector)
stock_sector_vol = stock_vol(stock_sector)


## 计算净值
sr_rtn = net_value(stock_size, stock_size_mom)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_size, stock_size_vol)
(sr_rtn + 1).cumprod().plot()

sr_rtn = net_value(stock_sector, stock_sector_mom, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_sector, stock_sector_vol, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()
