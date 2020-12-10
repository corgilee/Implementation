#https://en.wikipedia.org/wiki/Somers%27_D

import pandas as pd
import numpy as np
import time
from scipy.stats._stats import _kendall_dis

x=np.array([0.25]*4+[0.5]*12+[0.75]*8)
y=np.array([0]*3+[1]+[0]*5+[1]*7+[0]*2+[1]*6)

def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
            (cnt * (cnt - 1.) * (cnt - 2)).sum(),
            (cnt * (cnt - 1.) * (2*cnt + 5)).sum())

size = x.size
perm = np.argsort(y)  # sort on y and convert y to dense ranks
print("perm: ",perm)
x, y = x[perm], y[perm]

y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
print("transfomred y: ", y)

# stable sort on x and convert x to dense ranks
perm = np.argsort(x, kind='mergesort')
x, y = x[perm], y[perm]
x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

dis = _kendall_dis(x, y)  # discordant pairs

obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
xtie,x0, x1 = count_rank_tie(x)     # ties in x, stats
ytie,y0, y1 = count_rank_tie(y)     # ties in y, stats

print("xtie: ", xtie)

print("ytie: ", ytie)

# def SomersD(x, y):
#     x = np.asarray(x).ravel()
#     y = np.asarray(y).ravel()

#     if x.size != y.size:
#         raise ValueError("All inputs must be of the same size, "
#                          "found x-size %s and y-size %s" % (x.size, y.size))

#     def count_rank_tie(ranks):
#         cnt = np.bincount(ranks).astype('int64', copy=False)
#         cnt = cnt[cnt > 1]
#         return ((cnt * (cnt - 1) // 2).sum(),
#             (cnt * (cnt - 1.) * (cnt - 2)).sum(),
#             (cnt * (cnt - 1.) * (2*cnt + 5)).sum())

#     size = x.size
#     perm = np.argsort(y)  # sort on y and convert y to dense ranks
#     print("perm: ",perm)
#     x, y = x[perm], y[perm]

#     y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)
#     print("transfomred y: ", y)

#     # stable sort on x and convert x to dense ranks
#     perm = np.argsort(x, kind='mergesort')
#     x, y = x[perm], y[perm]
#     x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

#     dis = _kendall_dis(x, y)  # discordant pairs

#     obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
#     cnt = np.diff(np.where(obs)[0]).astype('int64', copy=False)

#     ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
#     xtie,x0, x1 = count_rank_tie(x)     # ties in x, stats
#     ytie,y0, y1 = count_rank_tie(y)     # ties in y, stats

#     tot = (size * (size - 1)) // 2

#     # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
#     #               = con + dis + xtie + ytie - ntie
#     #con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
#     SD = (tot - xtie - ytie + ntie - 2 * dis) / (tot - ntie)
#     return (SD, dis)

#con_plus_dis_plus_tie=tot- xtie- ytie + ntie

#print(SomersD(x,y))
# start_time = time.time()
# SD, dis = SomersD(df.realized_ead, df.pred_value)
# print("--- %s seconds ---" % (time.time() - start_time))

