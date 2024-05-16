'''
先把步骤讲出来，问问对方可不可以

1. Data exploration
    1. flag (positive rate, 一般可能是pos_rate 低的数据，后面要重点讨论 precision，recall，f1 score 和threshold，还有需不需要downsample）
    2. check_missing_rate (decide if some variables need to be dropped)
    3. data_type (numerical+categorical）
2. Feature engineering
    1. missing imputation (numerical data + categorical data)
    2. categorical data transformation
        1. check category high cardinal, if so, may use target encoding
    3. scale (optional)
3. Training data build up
    1. split
        1. 确认一下是否需要downsample negative 
4. Model training
    1. (先讲一讲， logistic regression, random forest , gbm tree 区别）
    2. model fit
    3. feature importance
    4. cross validation (optional)
'''

import pandas as pd
import numpy as np

### Data Type

df=pd.read_csv('data.csv')
df.info()
data.dtypes.value_counts()


# category/numerical
cat_features=df.select_dtypes(object).columns.tolist()

feature1=df.select_dtypes('float64').columns.tolist()
feature2=df.select_dtypes(int).columns.tolist()
num_features=feature1+feature2

# missing rate

percent_missing = \
pd.DataFrame(df.isnull().sum() * 100 / len(df)).reset_index()

percent_missing.columns=['columns','missing_rate']

#check the ratio of 50+ missing
sum(percent_missing['missing_rate']>=50)/percent_missing.shape[0]

#only keep the colums which has less than 50%
miss_50_minus=percent_missing.loc[percent_missing.missing_rate<50,'columns'].to_list()
df=df[miss_50_minus]

##### impuation
#impute numerical variables with median
for x in num_features:
    median_value=df[x].median()
    df[x]=df[x].fillna(median_value)

#impute cateogrical variables with "most frequent"
for x in cat_features:
    mode_value = df[x].mode()[0]
    df[x]=df[x].fillna(mode_value)

# category encoder
# option 1, ordinal encoder
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder() 
cat_features=['cat_0', 'cat_1', 'cat_2']
df[cat_features] = encoder.fit_transform(df[cat_features])
#--- or ----
df['cat1']=encoder.fit_transform(np.array(df['cat1']).reshape(-1,1))
df['cat1']=df['cat1'].astype('category') # 一定要把他 astype 成为 'categorical'

# target encoding
query_mean=pd.DataFrame(df.groupby(['query_text'])['has_product_click'].mean()).reset_index()
query_mean_dict=dict(zip(query_mean['query_text'],query_mean['has_product_click']))
df['query_text_encoding']=df['query_text'].map(query_mean_dict)


