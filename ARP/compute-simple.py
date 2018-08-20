def save_file(func):
    def wrapper(*args, **kw):
        arg0, *_ = args
        arg0_name = arg0.__name__
        func_name = func.__name__
        suffix = func_name.split(sep='_')[-1]
        price, position = func(*args, **kw)
        price.to_csv('.out/price/%s_%s.csv'%arg0_name%func_name)
        position.to_csv('.out/position/%s_%s.csv'%arg0_name%func_name)
        return 0
    print("Write Done!")
    return 0

@save_file
def stock_mom(df, m=5, lags=[210, 240, 270], positions=[0.5, 0.5, -0.5, -0.5]):
    df_factor = stock_mom_factor(df, m, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position

def stock_vol(df, m=5, lags=[210, 240, 270], positions=[0.5, 0.5, -0.5, -0.5]):
    df_factor = stock_vol_factor(df, m, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position

def bond_csm(df, lags=[20, 40, 60], positions=[0.5, 0.5, -0.5, -0.5]):
    df_factor = bond_csm_factor(df, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position

def bond_tsm(df, df_rf, lags=[20, 40, 60]):
    df_price, df_factor = bond_tsm_factor(df, df_rf, lags)
    df_position = bond_tsm_position(df_factor)
    return df_price, df_position

def bond_value(df, df_ytm, transform_matrix, m=5, lags=[50, 100, 150], positions=[0.5, 0.5, -0.5, -0.5]):
    ytm_transform = bond_value_ytm(df, df_ytm, transform_matrix)
    df_factor = bond_value_factor(ytm_transform, m, lags)
    df_position = factor_to_position(df_facotr, positions)
    df_price = df.copy()
    return df_price, df_factor

def bond_carry(df, df_rf, ytm_origin, transform_matrix, positions, m=60, x=[7/365, 2, 4, 6, 17/2, 50/3], lags=[250, 500, 750]):
    ytm_transform = bond_carry_ytm(df, df_rf, ytm_origin, transform_matrix)
    df_factor = bond_carry_facotr(ytm_transform, m, x, lags)
    df_position = bond_carry_position(df, df_carry, positions)
    df_price = df.copy()
    return df_price, df_position


def multi_rev(df, lags=[240, 300, 360], positions=[0.5, -0.5]):
    df_factor = multi_rev_facotr(df, lags)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position


def multi_vol(df, lags=[60, 120, 180], positions=[0.5, -0.5]):
    df_factor = multi_vol_factor(df, lags)
    df_position = factor_to_position(df_facotr, positions)
    df_price = df.copy()
    return df_price, df_position

def multi_value(df, df_value_stock, df_value_bond, m=5, positions=[0.5, -0.5]):
    df_factor = multi_value_factor(df, df_value_stock, df_value_bond, m)
    df_position = factor_to_position(df_factor, positions)
    df_price = df.copy()
    return df_price, df_position