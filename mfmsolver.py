# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:01:17 2018

@author: s_zhangyw
"""

def solver(w_1, w_bench, expect_rtn, trade_cost, X_style, X_industry, tol_style, tol_industry):
    
    ''' 在风格和行业中性约束下，求解最优权重
    如果资产数目为n，则

    Parameters
    ----------
    w_1 : array of shape (n)
        上期最优权重
    w_bench : array of shape (n)
        基准权重
    expect_rtn : array of shape (n)
        预测收益率
    trade_cost: float
        交易成本费率    
    X_style: array of shape (n)
        风格向量
    X_industry: array of shape (n, 行业类别数量)
        行业矩阵
    tol_style: float
        风格约束容忍度
    tol_industry: float
        行业约束容忍度

    Returns
    -------
    w* : array of shape (n)
        最优权重向量
    
    Notes
    -------
    这里假设风格因子为连续变量，如市值、市盈率等；
    如果为虚拟变量，则需参考行业约束设置约束条件。
    
    '''
    
    N = len(w)

    def func(w, w_1, expect_rtn, trade_cost, sign=-1.0):
        res = w.dot(expect_rtn) - trade_cost * (w - w_1).abs().sum() / 2.0
        return res * sign

    cons = ({'type': 'eq',
             'fun': lambda x: np.array(x.sum() - 1.0),
             'jac': lambda x: np.ones(N)},
             {'type': 'ineq',
             'fun': lambda x: tol_style - abs(x.dot(X_style) / w_bench.dot(X_style) - 1.0)},
            {'type': 'ineq',
             'fun': lambda x: np.ones(N) * tol_industry - (x - w_bench).dot(X_industry).abs()}
             )

    res = minimize(func, [1/N]*N, args=(w_1, w_bench, expect_rtn, trade_cost, X_style, X_industry, tol_style, tol_industry),
                   constraints=cons, bounds=[(0, 1)]*N, method='SLSQP', tol=1e-18, options={'disp': False, 'maxiter': 1000})

    return res.x

def main():
    
    ''' 如果资产数目为n，投资期数为q, 则

    需要输入:
    ----------
    arr_w_bench : array of shape (q, n)
        基准权重矩阵
    arr_expect_rtn : array of shape (q, n)
        预测收益率矩阵
    trade_cost: float
        交易成本费率    
    X_style: array of shape (n)
        风格向量
    X_industry: array of shape (n, 行业类别数量)
        行业矩阵
    tol_style: float
        风格约束容忍度
    tol_industry: float
        行业约束容忍度

    Returns
    -------
    arr_w* : array of shape (q, n)
        最优权重矩阵   
    '''
    
    arr_w_bench = 
    arr_expect_rtn = 
    trade_cost = 
    X_style = 
    X_industry = 
    tol_style = 
    tol_industry = 
    
    arr_w = arr_w_bench.copy()

    for i in range(len(arr_w)):
        if i = 0:
            arr_w[i] = arr_w_bench[i]
        else:
            arr_w[i] = solver(arr_w[i-1], arr_w_bench[i], arr_expect_rtn[i], trade_cost, X_style, X_industry, tol_style, tol_industry)

    return arr_w

if __name__ == '__main__':
    main()

