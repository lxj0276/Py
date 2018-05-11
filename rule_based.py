# -*- coding: utf-8 -*-
from scipy.stats import norm


def stop_loss(sr_ret, r_stop_loss, r_re_entry):
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
        sr_pos[i] = strategy(sr_pos[i-1], sr_ret[i])

    return sr_pos.shift(1)


def target_vol(sr_sigma, tar_sigma, MaxExp, tol_change):
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

    return sr_pos.shift(1)


def dd_control(sr_CVaR, tar_CVaR, MaxExp, tol_change):
    return target_vol(sr_CVaR, tar_CVaR, MaxExp, tol_change)


def CPPI(sr_ret_c, sr_ret_f, m, f0):

    def net_value(value, ret_c, ret_f):
        ratio_c = (value - f0) * m / (f0 + (value - f0) * m)
        return ratio_c, value * (1 + ret_c * ratio_c + ret_f * (1 - ratio_c))

    sr_ratio_c = sr_ret_c.copy()
    sr_net_value = sr_ret_c.copy()
    sr_ratio_c[0] = 1 - f0
    sr_net_value[0] = 1
    for i in range(1, len(sr_amt_c)):
        sr_ratio_c[i], sr_net_value[i] = net_value(
            sr_net_value[i-1], sr_ret_c[i-1], sr_ret_f[i-1])

    return sr_ratio


def CPPI(sr_ret_c, sr_ret_f, m, alpha):

    def net_value(value, ret_c, ret_f):
        ratio_c = value * (1 - alpha) * m / \
            (value * (1 - alpha) * m + value * alpha)
        return ratio_c, value * (1 + ret_c * ratio_c + ret_f * (1 - ratio_c))

    sr_ratio_c = sr_ret_c.copy()
    sr_net_value = sr_ret_c.copy()
    sr_ratio_c[0] = 1 - alpha
    sr_net_value[0] = 1
    for i in range(1, len(sr_amt_c)):
        sr_ratio_c[i], sr_net_value[i] = net_value(
            sr_net_value[i-1], sr_ret_c[i-1], sr_ret_f[i-1])

    return sr_ratio


def OBPI(sr_p_a, sr_p_b, sr_v_a, sr_v_b, t):
    sr_d1 = (np.log(sr_p_a / sr_p_b) + (sr_v_a - sr_v_b)
             ** 2 * t / 2) / ((sr_v_a - sr_v_b) * t ** 0.5)

    return sr_d1.apply(lambda d1: norm.cdf(d1))


def VaRcover(sr_rf, sr_VaR, p):
    return 1 / (p * sr_VaR / sr_rf + 1)


def margrabe(sr_p_a, sr_p_b, sr_v_a, sr_v_b, t):
    sr_pos = sr_p_a.copy()
    for i in range(len(sr_p_a)):
        d1 = (np.log(sr_p_a[i] / sr_p_b[i]) + (sr_v_a[i] - sr_v_b[i])
              ** 2 * t / 2) / ((sr_v_a[i] - sr_v_b[i]) * t ** 0.5)
        d2 = (np.log(sr_p_a[i] / sr_p_b[i]) - (sr_v_a[i] - sr_v_b[i])
              ** 2 * t / 2) / ((sr_v_a[i] - sr_v_b[i]) * t ** 0.5)
        sr_pos[i] = sr_p_a[i] / sr_p_b[i] * \
            norm.cdf(d1) / (sr_p_a[i] / sr_p_b[i] *
                            norm.cdf(d1) + 1 - norm.cdf(d2))

    return sr_pos
