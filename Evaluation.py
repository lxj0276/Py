# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:25:05 2018

@author: s_zhangyw
"""

import numpy as np
import pandas as pd

class evaluation(object):

    def __init__(self, df_rtn, df_pos, trade_cost=0):
        self.rtn = df_rtn
        self.pos = df_pos.apply(lambda x: x / sum(x) if sum(x), axis=1)
        self.trade_cost = trade_cost


    def net_value(self):
        sr_rtn = (df_pos * df_rtn).sum(axis=1)


        