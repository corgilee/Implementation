# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:41:10 2019

@author: zhiyu
"""

import pandas as pd
import numpy as np
d={'ID':['101','101','101','101','102'], 'Item':['iphone','ipad','iwatch','iphone','iphone']}

#d = {'col1': [1, 2], 'col2': [3, 4]}
df=pd.DataFrame(d)
print(df)

df1= df.groupby(['ID','Item'])['Item'].agg(['count']).reset_index()
print(df1)

''' 
#pivot_table 做法， 记住 index, columns
df1.pivot_table(index='ID',columns='Item')
print(df1)
'''

#items=list(df['Item'].unique())
nitems=df['Item'].nunique()
nids=df['ID'].nunique()
ids=list(df['ID'].unique())
items=list(df['Item'].unique())

df2=pd.DataFrame(np.zeros((nids,nitems)),columns=items)

def counter(item,ids):
    re=[]
    for id in ids:
        check=df1.loc[(df1.ID==id) & (df1.Item==item),['count']]
        if check.empty:
            re.append(0)
        else:
            re.append(check.values[0][0])
    return re
        
for item in items:
    df2[item]=counter(item, ids)
    
print(df2)