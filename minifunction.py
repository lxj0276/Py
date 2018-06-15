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


import seaborn as sns
sns.set()

sns.set(font='SimHei', style='ticks', font_scale=1.5, 
        palette=sns.color_palette('Set1', n_colors=13, desat=0.8))
plt.stackplot(pos_ew.index, pos_ew.values.T)

flatui = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", 
          "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#e6daa6", "#b15928",
          "#34495e", "#95a5a6"]
sns.set(font='SimHei', style='ticks', font_scale=1.5, 
        palette=sns.color_palette(flatui, n_colors=13, desat=0.8))


    