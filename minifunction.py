# -*- encoding: utf-8 -*-

# mini functions


# 获取函数参数名称
def get_args(funcname):
    argcount = eval(funcname + '.__code__.co_argcount')
    varnames = eval(funcname + '.__code__.co_varnames')

    return varnames[:argcount]


def re_sample(arr, lmd=2, seed=0):
    n = len(arr)
    np.random.seed(seed)
    x = np.random.exponential(scale=lmd, size=n)
    indices = np.array(list(map(int, n - n * x / x.max())))
    return arr[indices]



    