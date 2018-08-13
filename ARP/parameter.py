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
## 国债财富指数：中债-国债总财富(1-3年)指数   中债-国债总财富(3-5年)指数    中债-国债总财富(5-7年)指数    中债-国债总财富(7-10年)指数   中债-国债总财富(10年以上)指数
bond_treasury_index = ['CBA00621.CS', 'CBA00631.CS', 'CBA00641.CS', 'CBA00651.CS', 'CBA00661.CS']
## 金融债财富指数：中债-金融债券总财富(1-3年)指数   中债-金融债券总财富(3-5年)指数  中债-金融债券总财富(5-7年)指数  中债-金融债券总财富(7-10年)指数 中债-金融债券总财富(10年以上)指数
bond_finance_index = ['CBA01221.CS', 'CBA01231.CS', 'CBA01241.CS', 'CBA01251.CS', 'CBA01261.CS']
## AAA企业债财富指数：中债-企业债AAA财富(1-3年)指数   中债-企业债AAA财富(3-5年)指数 中债-企业债AAA财富(5-7年)指数 中债-企业债AAA财富(7-10年)指数    中债-企业债AAA财富(10年以上)指数
bond_corporate_index = ['CBA04221.CS', 'CBA04231.CS', 'CBA04241.CS', 'CBA04251.CS', 'CBA04261.CS']
## 债券YTM数据：国债2年 国债4年    国债6年    国债8年    国债9年    国债10年   国债20年
## 国开债2年   国开债4年   国开债6年   国开债8年   国开债9年   国开债10年  国开债20年
## 企业债2年   企业债4年   企业债6年   企业债8年   企业债9年   企业债10年  企业债15年



## 跨类指数：中证全指  恒生中国企业指数  中债-国债总财富(总值)指数 中债-金融债券总财富(总值)指数 中债-企业债AAA财富(总值)指数
asset_composite_index = ['000985.CSI', 'HSCEI.HI', 'CBA00601.CS', 'CBA01201.CS', 'CBA04201.CS']


## 要写的函数
## vol mom 

# 股票mom算法
# 计算逻辑：最近五日均值与历史L天最大值之比，最后再取平均
def stock_mom(df, m=5, l=[210, 240, 270]):
    def mom(lag):
        return df.rolling(m).mean() / df.rolling(lag).max().shift(1) - 1
    
    return mom(l[0]) + mom(l[1]) + mom(l[2])

# 股票vol算法
# 计算逻辑：最近五日波动率均值与L天之前的五日均值之比，最后取平均
def stock_vol(df, m=5, l=[240, 270, 300]):
    log_rtn = np.log(df).diff(1)
    std = log_rtn.expanding().std()
    def vol(lag):
        return std.rolling(m).mean() / std.rolling(m).mean().shift(lag)
    return vol(l[0]) + vol(l[1]) + vol(l[2])

# 债券csm算法
# 计算逻辑：L天的收益率，取平均
def bond_csm(df, l=[20, 40, 60]):
    def csm(lag):
        return df.pct_change(lag)
    return csm(l[0]) + csm(l[1]) + csm(l[2])

# 债券tsm算法
# 计算逻辑：

# 债券value算法
# 计算逻辑：YTM五日平均，分别取50，100，150日最大值的比率平均
def bond_value(df, m=5):
    pass


# 债券carry算法
# 计算逻辑：YTM十日平均，对一组数求回归斜率，（最近10日平均-最近150日平均） * 100
def bond_carry(df):
    pass

## 跨类 rev vol value
# 跨类rev算法
## 计算逻辑：计算对数收益率，某段历史正收益之和与负收益之和的比值，取负，取对数，再取负
## 简化之后，负收益之和与正收益之和的比值，取绝对值，取对数
def asset_rev(df, l=[240, 300, 360]):
    log_rtn = np.log(df).diff(1)
    def rev(lag):
        return log_rtn.rolling(lag).apply(lambda x: np.log(-np.sum(x[x<0]) / np.sum(x[x>0])))
    return rev(l[0]) + rev(l[1]) + rev(l[2])

# 跨类vol算法
## 计算逻辑：对数收益率，三段标准差的均值，三段区间标准差的变化率，再取均值
def asset_vol(df, l=[60, 120, 180]):
    log_rtn = np.log(df).diff(1)
    def std(lag):
        return log_rtn.rolling(lag).std()
    std = std(lag[0]) + std(lag[1]) + std(lag[2])
    def vol(lag):
        return std.diff(lag)
    return vol(lag[0]) + vol(lag[1]) + vol(lag[2])


# 跨类value算法
## 计算逻辑：对于两个股票指数，100 / pe, pb, ps, 最近五日平均，历史score, 再取平均，对于债券指数，YTM五日平均，取score
def asset_value(df, m=5):
    pass


# 先开始做吧
## 要写的函数
## vol mom 

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


stock_size = pd.read_csv('stock_size.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_size_mom = stock_mom(stock_size)
stock_size_vol = stock_vol(stock_size)

sr_rtn = net_value(stock_size, stock_size_mom)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_size, stock_size_vol)
(sr_rtn + 1).cumprod().plot()


stock_sector = pd.read_csv('stock_sector.csv', index_col=[0], header=[0, 1], parse_dates=[0])
stock_sector_mom = stock_mom(stock_sector)
stock_sector_vol = stock_vol(stock_sector)

sr_rtn = net_value(stock_sector, stock_sector_mom, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()
sr_rtn = net_value(stock_sector, stock_sector_vol, [1/14]*14 + [-1/14]*14)
(sr_rtn + 1).cumprod().plot()


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
def bond_tsm(df, df_rf, l=[20, 40, 60]):
    df_rf.columns = "rf"
    df_bind = pd.concat([df, df_rf], axis=1)
    def tsm(lag):
        return df.pct_change(lag)
    df_tsm = (tsm(l[0]) + tsm(l[1]) + tsm(l[2])).shift(1)
    return df_tsm

def bond_tsm_pos(df_tsm):
    df = df_tsm - df_tsm.rf
    df = df.applymap(lambda x: 0 if x < 0 else 1)
    df = df / df.sum(axis=1)
    return df

# 债券value算法
# 计算逻辑：YTM五日平均，分别取50，100，150日最大值的比率平均
def bond_value(df, df_ytm, m=5, l=[50, 100, 150]):
    df_avg = df_ytm.rolling(m).mean()
    def value(lag):
        df = df_avg / df_acg.rolling(lag).max()
        return df
    df_value = value(l[0]) + value(l[1]) + value(l[2])
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


def bond_carry(df, df_ytm, m=60, l=[250, 500, 750]):
    x = [7/365, 2, 4, 6, 8.5, 50/3]
    df_slope = df_ytm.apply(lambda y: (x - x.mean()).dot(y - y.mean()) / (x - x.mean()).dot(x - x.mean()))
    def carry(lag):
        df = (df_slope.rolling(m).mean() - df.shift(lag).rolling(m).mean()) * 100
        return df
    df_carry = carry(l[0]) + carry(l[1]) + carry(l[2])
    return df_carry

def bond_carry_pos(df, df_carry):
    pos_list = long_short(df)
    TODO 


def long_short(df):
    num_assets = df.shape[1]
    num_half = num_assets // 2
    pos_list = [0] * num_assets
    pos_list[:num_half] = 1 / num_half
    pos_list[-num_half:] = -1 / num_half
    return pos_list

