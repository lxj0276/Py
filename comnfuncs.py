def clear(arr, d=2):
    brr = arr.copy()
    for i in range(len(arr)):
        brr[i] = round(arr[i], d)
    brr[brr.argmax()] = brr[brr.argmax()] + (1.0 - brr.sum())

    return brr

# z为净值list，f=20代表数据采集间隔为20个交易日
def indics(z, f=20):
    N = len(z)
    rtn = (z[-1] / z[0]) ** (250 / f / N) - 1
    vol = z.pct_change().std() * ((250 / f) ** 0.5)
    sp = (rtn - 0.04) / vol
    dd = []
    for i in range(N):
        Z = z[:i+1]
        dd.append(1 - Z[-1] / Z.max())
    dd_max = max(dd)
    cm = rtn / dd_max

    return rtn, vol, sp, dd_max, cm
