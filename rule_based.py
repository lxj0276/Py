# -*- coding: utf-8 -*-

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
    def startegy(pos, sigma):
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

def 

