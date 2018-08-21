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