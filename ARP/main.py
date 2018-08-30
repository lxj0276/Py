# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:08:39 2018

@author: zhangyw49
"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from update_data import update_data
from compute_position import compute_position
from compute_return import compute_return
from compute_portfolio import update_portfolio, compute_portfolio, file_to_frame
from evaluation import evaluation

# 第一步：更新数据
update_data()
# 第二步：计算单个ARP仓位
compute_position()
# 第三步：计算单个ARP收益
compute_return()
# 第四步：在单个APR基础上，构建投资组合。
# 注：compute_portfolio可进行全部运算，update_portfolio之进行增量运算
#update_portfolio()
compute_portfolio()



# =============================================================================
#  对回测结果进行分析
# =============================================================================
name_list = ['EW', 'MV', 'EMV', 'RP', 'RB', 'MD', 'DeCorr', 'MSR', 'MVO', 'ReSmp']
price = file_to_frame("./out/portfolio/price/price_bind.csv")
rtn_list = []
position_list = []
for name in name_list:
    rtn = file_to_frame("./out/portfolio/return/%s.csv"%name, header=None)
    position = file_to_frame("./out/portfolio/position/%s.csv"%name)
    rtn_list.append(rtn)
    position_list.append(position)  
## 作图
value_list = list(map(lambda x: (1 + x).cumprod(), rtn_list))
plt.figure(dpi=100)
for value in value_list:
    plt.plot(value)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(name_list, fontsize=12, bbox_to_anchor=(1, 0.5), loc=6)
## 评价指标
df_rtn = price.pct_change(1)
for pos in position_list:
    E = evaluation(pos.dropna(), df_rtn['2010':])
    print(E.YearRet())
for pos in position_list:
    E = evaluation(pos.dropna(), df_rtn['2010':])
    print('####', E.CAGR(), E.Ret_Roll_1Y()[0], E.Ret_Roll_3Y()[0], E.Stdev(), E.Skew(), E.MaxDD(), E.MaxDD_Dur(), 
          E.VaR(), E.SR(), E.Calmar(), E.RoVaR(), E.Hit_Rate(), E.Gain2Pain(), sep='\n')
