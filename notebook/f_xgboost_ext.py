### Data Loading and Initial Exploration

import numpy as np 
import pandas as pd 
import csv 
data = pd.read_csv("faire-ml-rank-small.csv") 
data.shape

# Get a summary of the DataFrame including data types, non-null values, and memory usage
data.info()
#data.columns.tolist() 

print("--- Target Variable Distribution ('has_product_click') ---") 
click_counts = data['has_product_click'].value_counts() 
print(click_counts) 
print(f"Click rate: {click_counts[1] / len(data) * 100 : .2f}%") 

#Feature Engineering: Encoding and Missing Value Imputation

# --- Numerical Missing Value Imputation ---
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns 
num_cols_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0] 
# 先print 看看
print(num_cols_with_missing)

high_missing_threshold = 0.75 
for col in num_cols_with_missing: 
    missing_rate = data[col].isnull().mean() 
    if missing_rate > high_missing_threshold: 
        # Strategy: Binary Indicator + Fill with 0
        '''
        The core reason for using this strategy is to preserve the predictive information 
        contained in both the value of the feature (when present) and the fact that it was missing (when absent).
        '''
        
        data[f'{col}_is_known'] = data[col].notnull().astype(int) 
        data[col] = data[col].fillna(0) 
    else: 
        # Strategy: Median Imputation
        median_val = data[col].median() 
        data[col] = data[col].fillna(median_val)


print("--- Object Type Categorical Features ---") 
object_cols = data.select_dtypes(include='object') 
for col in object_cols.columns: 
    not_null_count = data[col].count() 
    unique_count = data[col].nunique() 
    print(f"'{col}': {not_null_count : <14} | {unique_count : <13} unique values")


# xgboost
from sklearn.model_selection import train_test_split 
import xgboost as xgb 
from sklearn.metrics import roc_auc_score

# === 1. Build Training Data (Feature Selection) ===
y = data['has_product_click'] 
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist() 
id_cols_to_drop = ['has_product_click', 'request_id_anon', 'retailer_token_anon'] 
numerical_features = [col for col in numeric_cols if col not in id_cols_to_drop] 
X_numeric = data[numerical_features] 

# One-Hot Encoding for 'fillter_string_id'
# 生成新的列 'fillter_string_id'
# factorize() 返回一个 tuple：第一个元素是整数编码的数组，第二个元素是唯一的类别列表
data['fillter_string_id'] = data['filter_string'].factorize()[0]

filter_dummies = pd.get_dummies(data['fillter_string_id'], prefix='filter_string') 
X = pd.concat([X_numeric, filter_dummies], axis=1)

# === 2. Split Data (Train / Validation) ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 


import xgboost as xgb
# --- 1. 准备 DMatrix ---
# 原生 XGBoost 训练需要 DMatrix 格式的数据
# 假设 X_train, y_train, X_val, y_val 已经定义
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

# --- 2. 定义参数 (使用 Baseline 参数) ---
params = {
    'objective': 'binary:logistic',  # 目标函数：预测点击概率
    'eval_metric': 'logloss',        # 评估指标
    'eta': 0.1,                      # 学习率 (对应 learning_rate)
    'max_depth': 5,                  # 树的深度
    'seed': 42                       # 随机种子 (对应 random_state)
    # 不平衡数据可以在这里设置 scale_pos_weight
}

# 使用 watchlist 监控训练和验证集
watchlist = [(dtrain, 'train'), (dval, 'eval')]

model= xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,           # 迭代次数 (对应 n_estimators)
    evals=watchlist,               # 传入监控集
    verbose_eval=10                # 每 10 轮打印一次结果
)

## next version
params = { 
    'objective': 'binary:logistic', 
    'eval_metric': 'logloss', 
    'eta': 0.05, 
    'max_depth': 4, 
    'reg_alpha': 0.05, 
    'reg_lambda': 5, 
    'tree_method': 'hist', 
    'seed': 42
} 

watchlist = [(dtrain, 'train'), (dval, 'eval')] 
model = xgb.train(
    params=params, 
    dtrain=dtrain, 
    num_boost_round=500, 
    evals=watchlist, 
    early_stopping_rounds=10, 
    verbose_eval=10
)

# === 4. Evaluation (ROC AUC) ===
dtrain_eval = xgb.DMatrix(X_train) 
dval_eval = xgb.DMatrix(X_val) 
y_pred_proba_val = model.predict(dval_eval) 
y_pred_proba_train = model.predict(dtrain_eval) 

auc_val = roc_auc_score(y_val, y_pred_proba_val) 
auc_train = roc_auc_score(y_train, y_pred_proba_train) 
print(f"Training ROC AUC: {auc_train : .4f}") 
print(f"Validation ROC AUC: {auc_val : .4f}") 


# Feature Importance
importance = model.get_score(importance_type='gain') 
feature_importances = pd.Series(importance).sort_values(ascending=False) 
print("--- Top 5 Important Features (by Gain) ---") 
print(feature_importances.head(5))


### Code to Implement These 2 Features
def check_query_in_title(row):
    """
    检查搜索词 (query_text) 是否包含在商品标题 (title) 中，不区分大小写。
    返回 1 (包含) 或 0 (不包含)。
    """
    query = row['query_text'].lower()
    title = row['title'].lower()
    if query in title:
        return 1
    else:
        return 0

# 应用函数到 DataFrame 的每一行 (axis=1)
data['title_has_query'] = data.apply(check_query_in_title, axis=1)

# === Feature 2: Days Since Creation (Recency) ===
# FIX: 添加 format='mixed' 来同时处理带毫秒和不带毫秒的时间格式
data['created_at_dt'] = pd.to_datetime(data['created_at_a'], format='mixed')

# Define a reference date (using the max date in the dataset)
current_date = data['created_at_dt'].max()

# Calculate the difference in days
data['days_since_created'] = (current_date - data['created_at_dt']).dt.days


### grid search if necessary
# Define the parameter grid to search

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import time # Optional: to track how long the search takes

param_grid = {
    'max_depth': [3, 4, 5], 
    'learning_rate': [0.05, 0.1, 0.2], 
    'n_estimators': [100, 200, 300], # Number of boosting rounds
    'reg_alpha': [0.0, 0.05, 0.1],   # L1 regularization
    'reg_lambda': [1, 5, 10]        # L2 regularization
}

# The total number of runs will be 3 * 3 * 3 * 3 * 3 = 243 models * CV folds
print(f"Total models to train: {3*3*3*3*3} * CV folds")

xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    seed=42,
    # Add scale_pos_weight if you want to explicitly handle the 4.2% click rate
    # scale_pos_weight= (len(y_train) - y_train.sum()) / y_train.sum()
)

# Initialize the GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc', # Metric to optimize
    cv=3,              # 3-fold cross-validation
    verbose=1,         # Print progress
    n_jobs=-1          # Use all available CPU cores
)

'''
# Run the search (this may take a long time!)
print("\nStarting Grid Search...")
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()
print(f"Grid Search Complete in {end_time - start_time : .2f} seconds.")

# --- Results ---
print("\n## Grid Search Results ##")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation ROC AUC: {grid_search.best_score_ : .4f}")

# Final validation on the held-out set
best_model = grid_search.best_estimator_
y_pred_proba_val = best_model.predict_proba(X_val)[:, 1]
final_auc = roc_auc_score(y_val, y_pred_proba_val)
print(f"Validation ROC AUC (Best Model): {final_auc : .4f}")
'''



### target encoding
### Optional
# === 2. Non-Leaky Target Encoding (TE) for "query_text" ===

# 1. 计算全局平均点击率 (仅使用训练集 y_train)
global_mean = y_train.mean()

# 2. 计算 TE Map (仅使用训练集 X_train 和 y_train)
# 将 X_train 和 y_train 合并，计算每个 query_text 的平均点击率
'''

te_mapping = pd.Series(y_train.values, index=X_train['query_text']).groupby(level=0).mean()

X_train['query_text_te'] = X_train['query_text'].map(te_mapping).fillna(global_mean)
X_val['query_text_te'] = X_val['query_text'].map(te_mapping).fillna(global_mean)

print("--- Target Encoding Complete (Non-Leaky) ---")

'''



