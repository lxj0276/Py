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
    dd_max=max(dd)
    cm = rtn / dd_max
    print(rtn, vol, sp, dd_max, cm)
    
    return rtn, vol, sp, dd_max, cm

def nvfig(z, title):
    plt.figure(dpi=300)
    #plt.figure(figsize=(15,8))
    plt.plot(z)
    plt.xlabel('年份')
    plt.ylabel('净值')
    plt.title(title)
    plt.savefig('C:/Users/tober/Desktop/%s.png'%title)
    return 

def wtfig(z, w, title, clmn=df_return.columns):
    df_w=pd.DataFrame(index=z.index, columns=clmn, data=w)
    plt.figure(dpi=300)    
    #plt.figure(figsize=(15,8))
    plt.stackplot(df_w.index, df_w.T.values)
    plt.legend(clmn)
    plt.xlabel('年份')
    plt.ylabel('权重')
    plt.title(title)
    plt.savefig('C:/Users/tober/Desktop/%s_wt.png'%title)
    return

def fig(z, w, title):
    nvfig(z, title)
    wtfig(z, w, title)
    return 



