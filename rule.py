# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm


def stop_loss(sr_ret, r_stop_loss, r_re_entry):
    """
    止损

    Parameters
    ----------
    sr_ret: Series, 收益率序列
    r_stop_loss: float, 止损线
    r_re_entry: float, 入场线

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    def strategy(pos, ret):
        if ret > r_re_entry:
            return 1
        elif r_stop_loss <= ret <= r_re_entry:
            return pos
        else:
            return 0

    sr_pos = sr_ret.copy()
    sr_pos[0] = 0
    for i in range(1, len(sr_ret)):
        sr_pos[i] = strategy(sr_pos[i-1], sr_ret[i-1])

    return sr_pos


def stop_loss_dynamic(sr_ret, q_stop_loss, q_re_entry):
    """
    止损

    Parameters
    ----------
    sr_ret: Series, 收益率序列
    r_stop_loss: float, 止损线
    r_re_entry: float, 入场线

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    def strategy(pos, ret, r_stop_loss, r_re_entry):
        if ret > r_re_entry:
            return 1
        elif r_stop_loss <= ret <= r_re_entry:
            return pos
        else:
            return 0

    sr_pos = sr_ret.copy()
    sr_pos[0] = 0
    for i in range(1, len(sr_ret)):
        r_stop_loss = sr_ret[:i].quantile(q_stop_loss)
        r_re_entry = sr_ret[:i].quantile(q_re_entry)
        sr_pos[i] = strategy(sr_pos[i-1], sr_ret[i-1], r_stop_loss, r_re_entry)

    return sr_pos

def target_vol(sr_sigma, tar_sigma, MaxExp, tol_change):
    """
    目标波动率

    Parameters
    ----------
    sr_sigma: Series, 预测波动率
    tar_sigma: float, 目标波动率
    MaxExp: float, 风险敞口上限
    tol_change： float, 仓位变动幅度

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    def strategy(pos, sigma):
        tar_w = min(tar_sigma / sigma, MaxExp)
        if (1 - tol_change) < (pos / tar_w) < (1 + tol_change):
            return pos
        else:
            return tar_w

    sr_pos = sr_sigma.copy()
    sr_pos[0] = min(tar_sigma / sr_sigma[0], MaxExp)
    for i in range(1, len(sr_sigma)):
        sr_pos[i] = strategy(sr_pos[i-1], sr_sigma[i])

    return sr_pos


def target_vol_dynamic(sr_sigma, q_tar_sigma, MaxExp, tol_change):
    """
    目标波动率

    Parameters
    ----------
    sr_sigma: Series, 预测波动率
    tar_sigma: float, 目标波动率
    MaxExp: float, 风险敞口上限
    tol_change： float, 仓位变动幅度

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    def strategy(pos, sigma):
        tar_w = min(tar_sigma / sigma, MaxExp)
        if (1 - tol_change) < (pos / tar_w) < (1 + tol_change):
            return pos
        else:
            return tar_w

    sr_pos = sr_sigma.copy()
    sr_pos[0] = MaxExp
    for i in range(1, len(sr_sigma)):
        tar_sigma = sr_sigma[:i].quantile(q_tar_sigma)
        sr_pos[i] = strategy(sr_pos[i-1], sr_sigma[i])

    return sr_pos


def dd_control(sr_CVaR, tar_CVaR, MaxExp):
    """
    回撤控制

    Parameters
    ----------
    sr_CVaR: Series, 预测CVaR
    tar_CVaR: float, 目标CVaR
    MaxExp: float, 风险敞口上限

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    sr = sr_CVaR.copy()
    sr[sr>=0] = -1e-8  # 避免CVaR为正的情况
    return (tar_CVaR / sr).clip(0, MaxExp)


def CPPI(sr_ret, m, f0):
    """
    CPPI

    Parameters
    ----------
    sr_ret_c: Series, 风险资产收益序列
    m: 乘数
    f0: float, 无风险资产初始比例

    Returns
    -------
    sr_ratio: Series, 风险资产仓位序列
    """

    def net_value(value, ret):
        ratio = (value - f0) * m / (f0 + (value - f0) * m)
        return ratio, value * (1 + ret * ratio)

    sr_ratio = sr_ret.copy()
    sr_net_value = sr_ret.copy()
    sr_ratio[0] = 1 - f0
    sr_net_value[0] = 1
    for i in range(1, len(sr_ratio)):
        sr_ratio[i], sr_net_value[i] = net_value(
            sr_net_value[i-1], sr_ret[i-1])

    return sr_ratio


def TIPP(sr_ret, m, alpha):
    """
    复制型TIPP

    Parameters
    ----------
    sr_ret: Series, 风险资产收益序列
    m: float, 乘数
    alpha: float, 保本比例

    Returns
    -------
    sr_ratio: Series, 风险资产仓位序列
    """
    def net_value(value, ret):
        value = max(value, alpha)    ## alpha也是最低保本额度
        ratio = value * (1 - alpha) * m / \
            (value * (1 - alpha) * m + value * alpha)
        return ratio, value * (1 + ret * ratio)

    sr_ratio = sr_ret.copy()
    sr_net_value = sr_ret.copy()
    sr_ratio[0] = 1 - alpha
    sr_net_value[0] = 1
    for i in range(1, len(sr_ratio)):
        sr_ratio[i], sr_net_value[i] = net_value(
            sr_net_value[i-1], sr_ret[i-1])

    return sr_ratio


def OBPI(sr_p, sr_sigma, t):
    """
    复制型OBPI

    Parameters
    ----------
    sr_p: Series, 资产价格序列
    sr_sigma: Series, 资产波动率序列
    t: float, 时间

    Returns
    -------
    sr_pos: Series, 仓位序列
    """
    sr_d1 = (np.log(sr_p) + sr_sigma **
             2 * t / 2) / (sr_sigma * t ** 0.5)

    return sr_d1.apply(lambda d1: norm.cdf(d1))


def VaRcover(sr_VaR, rf, p):
    """
    VaR套补

    Parameters
    ----------
    sr_VaR: Series, VaR序列
    rf: float, 无风险收益率
    p: float, 套补比例

    Returns
    -------
    sr_pos: Series, 仓位序列
    """

    return 1.0 / (p * sr_VaR / rf + 1)


def margrabe(sr_p, sr_sigma, t):
    """
    margrabe资产交换

    Parameters
    ----------
    sr_p: Series, 资产价格序列
    sr_sigma: Series, 资产波动率序列
    t: float, 时间

    Returns
    -------
    sr_pos: Series, a资产仓位序列
    """
    sr_pos = sr_p.copy()
    for i in range(len(sr_p)):
        d1 = (np.log(sr_p[i]) + sr_sigma[i]
              ** 2 * t / 2) / (sr_sigma[i] * t ** 0.5)
        d2 = (np.log(sr_p[i]) - sr_sigma[i]
              ** 2 * t / 2) / (sr_sigma[i] * t ** 0.5)
        sr_pos[i] = sr_p[i] * \
            norm.cdf(d1) / (sr_p[i] *
                            norm.cdf(d1) + 1 - norm.cdf(d2))

    return sr_pos
