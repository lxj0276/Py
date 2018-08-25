# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:08:39 2018

@author: zhangyw49
"""

from update_data import update_data
from compute_position import compute_position
from compute_return import compute_return
from compute_portfolio import update_portfolio

# 更新数据
update_data()
# 计算单个ARP仓位
compute_position()
# 计算单个ARP收益
compute_return()
# 在单个APR基础上，构建投资组合，
# compute_portfolio可进行全部运算，update_portfolio之进行增量运算
update_portfolio()



