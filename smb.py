# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:16:59 2018

@author: s_zhangyw
"""
import seaborn as sns
sns.set(font='SimHei', color_codes=True)
file_s = 'smb_s.csv'
file_b = 'smb_b.csv'

df_s = pd.read_csv(file_s, encoding="GBK", index_col=0, parse_dates=[0])
df_b = pd.read_csv(file_b, encoding="GBK", index_col=0, parse_dates=[0])

sr = (df_s.iloc[:, -1] - df_b.iloc[:, -1]).dropna() / 100.0
(1 + sr).cumprod().plot()

# k日动量
k=60
#sr_1 = (df_s.iloc[:, -1]).diff(1) - (df_b.iloc[:, -1]).diff(1)
#sr_mom = sr_1.rolling(k).std().rolling(k).mean()
sr_mom = sr.dropna().rolling(k).apply(lambda x: reg_beta(x))
sr_mom.name='Momentum'
# h日持有收益
h=10
sr_ret = sr.rolling(h).apply(lambda x: (1 + x).prod() - 1).shift(-h)
sr_ret[sr_ret>0] = 1
sr_ret[sr_ret<0] = 0
sr_ret.name='Return'

sns.regplot(x=sr_mom[-1200:], y=sr_ret[-1200:], ci=None);

sns.regplot(x=sr_mom[-1200:], y=sr_ret[-1200:], ci=None, logistic=True);



# 动量之差
df_s = pd.read_csv(file_s, encoding="GBK", index_col=0, parse_dates=[0])
df_b = pd.read_csv(file_b, encoding="GBK", index_col=0, parse_dates=[0])

sr = (df_s.iloc[:, -1] - df_b.iloc[:, -1]).dropna() / 100.0
(1 + sr).cumprod().plot()

sr_s = (df_s.iloc[:, -1]).dropna() / 100.0
sr_b = (df_b.iloc[:, -1]).dropna() / 100.0

k = 20
sr_mom  = sr_s.rolling(k).std() - sr_b.rolling(k).std()

sns.regplot(x=sr_mom[-3200:], y=sr_ret[-3200:], ci=None);

sns.regplot(x=sr_mom[-3200:], y=sr_ret[-3200:], ci=None, logistic=True);
# 波动率之差





def reg_beta(arr):
    X = X = np.vstack([np.ones(len(arr)-1), arr[:-1]])
    _, beta = np.linalg.lstsq(X.T, arr[1:], rcond=None)[0]
    return beta

## 动量驱动
sr_pos = sr_mom.apply(lambda x: 1 if x>0 else -1)
(sr_pos[::h] * sr_ret[::h]+ 1).cumprod().plot()

f = open('print.log', 'w+')
# k日动量
for k, h in [(i, j) for i in range(1, 100, 5) for j in range(1, 60, 2)]:
    sr_mom = sr.rolling(k).apply(lambda x: (1 + x).prod() - 1)[10:]
    sr_ret = sr.rolling(h).apply(lambda x: (1 + x).prod() - 1).shift(-h)[10:]
    print(k, h, ":", sr_mom.corr(sr_ret,method='kendall'), '\n#####################', file=f)
          
    
          
          
df_s = pd.read_csv(file_s, encoding="GBK", index_col=0, parse_dates=[0])
df_b = pd.read_csv(file_b, encoding="GBK", index_col=0, parse_dates=[0])

sr = (df_b.iloc[:, -1]).dropna() / 100.0
(1 + sr).cumprod().plot()

# k日动量
k=360
#sr_1 = (df_s.iloc[:, -1]).diff(1) - (df_b.iloc[:, -1]).diff(1)
#sr_mom = sr_1.rolling(k).std().rolling(k).mean()
sr_mom = sr.rolling(k).apply(lambda x: (1 + x).prod() - 1)
sr_mom.name='Momentum'
# h日持有收益
h=120
sr_ret = sr.rolling(h).apply(lambda x: (1 + x).prod() - 1).shift(-h)
#sr_ret[sr_ret>0] = 1
#sr_ret[sr_ret<0] = 0
sr_ret.name='Return'

plt.scatter(sr_mom[-1000:], sr_ret[-1000:])

sns.regplot(x=sr_mom[-1000:], y=sr_ret[-1000:], marker='.', ci=None)

sns.regplot(x=sr_mom[-1000:], y=sr_ret[-1000:], marker='.', ci=None, logistic=True)


sr_mom = sr.rolling(k).std()

f = open('print.log', 'w+')
# k日动量
for k, h in [(i, j) for i in range(1, 240, 5) for j in range(1, 120, 2)]:
    sr_mom = sr.rolling(k).sum()
    sr_ret = sr.rolling(h).apply(lambda x: (1 + x).prod() - 1).shift(-h)
    print(k, h, ":", sr_mom.corr(sr_ret,method='kendall'), '\n#####################', file=f)
          
 