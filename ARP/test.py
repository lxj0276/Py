# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:08:39 2018

@author: zhangyw49
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()
path = "./out/single/return/"

fs = os.listdir(path)
for f in fs:
    r = pd.read_csv(path + f, index_col=[0], header=None)
    print(r.sum())



    plt.plot((1 + r).cumprod())
    plt.show()

