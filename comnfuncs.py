def clear(arr, d=2):
    brr = arr.copy()
    for i in range(len(arr)):
        brr[i] = round(arr[i], d)
    brr[brr.argmax()] = brr[brr.argmax()] + (1.0 - brr.sum())

    return brr


def indics(z):
    N = len(z)
    rtn = (z[-1] / z[0]) ** (12 / N) - 1
    vol = z.pct_change().std() * (12 ** 0.5)
    sp = (rtn - 0.04) / vol
    dd = []
    for i in range(N):
        Z = z[:i+1]
        dd.append(1 - Z[-1] / Z.max())
    dd_max = max(dd)
    cm = rtn / dd_max

    return rtn, vol, sp, dd_max, cm
