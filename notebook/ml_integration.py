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
    5. resample the training data (downsample the negative sample, optional)
'''

import pandas as pd
import numpy as np

### Data Type

df=pd.read_csv('data.csv')
df.info()
data.dtypes.value_counts()


# category/numerical
cat_features=df.select_dtypes(object).columns.tolist()

num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


# missing rate
percent_missing = (df.isnull().sum()*100/len(df)).reset_index(name="missing_rate")\
      .query("missing_rate > 0")\
      .sort_values("missing_rate", ascending=False)\
      .reset_index(drop=True)


#check the ratio of 50+ missing
sum(percent_missing['missing_rate']>=50)/percent_missing.shape[0]

#only keep the colums which has less than 50%
miss_50_minus=percent_missing.loc[percent_missing.missing_rate<50,'columns'].to_list()
df=df[miss_50_minus]

##### imputation
#impute numerical variables with median, 也可以提一嘴，在split 之后做imputation 其实能更好的预防leakage
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

# optino 2: target encoding
query_mean=pd.DataFrame(df.groupby(['query_text'])['has_product_click'].mean()).reset_index()
query_mean_dict=dict(zip(query_mean['query_text'],query_mean['has_product_click']))
df['query_text_encoding']=df['query_text'].map(query_mean_dict)

# option3, 用pd.get_dummies
categorical_vars = ['home_ownership']
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
# how to pick the newly generated column
new_dummy_cols = [c for c in df.columns if c not in cols_before]

#option 4: one hot encoder (比较麻烦)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop=None)
df_encoded = encoder.fit_transform(df[['Color']])

### time transfer ###
df["event_ts"] = pd.to_datetime(df["event_ts_str"],errors="coerce")

### textual preprocessing ##
## if lower case of the string constains a certain word like "exam"
df["text_has_exam"] = (
    df["text_column"].str.lower().str.contains("exam", na=False).astype(int)
)

## if col 1 (string format) are included in col 2 (string format)
df["col1_in_col2"] = (
    df.apply(lambda row: str(row["col1"]).lower() in str(row["col2"]).lower(),
             axis=1)
      .astype(int)
)


#scaling
from sklearn.preprocessing import StandardScaler

#standard 
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)



##Modeling
x=df[num_features].drop(columns=['label'])
y=df['label']
#--- Split -----
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=78,stratify=y)

# split by time
cutoff = df["event_ts"].quantile(0.8)
train = df[df["event_ts"] <= cutoff]
val = df[df["event_ts"] > cutoff]


### if negative sample undersampling is needed
train_df = x_train.copy()
train_df["y"] = y_train

pos = train_df[train_df.y == 1]
neg = train_df[train_df.y == 0].sample(n=len(pos) * 20, random_state=78)
train_df_balanced = pd.concat([pos, neg])

x_train_balanced = train_df_balanced.drop(columns=["y"])
y_train_balanced = train_df_balanced["y"]
##########

#### imputation ####
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(x_train)
x_val  = imputer.transform(x_val)


# Training
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(n_estimators=100,
max_depth=3,random_state=78,verbose=-1,subsample=0.8,colsample_bytree=0.8,min_child_samples=5)

## more regularization
'''
min_child_samples=50,
reg_alpha=1.0, 
reg_lambda=5.0,
'''

lgb_model.fit(x_train.values,y_train,eval_set=[(x_val.values,y_val)],eval_metric='average_precision',
              categorical_feature=[23],callbacks=[lgb.early_stopping(10)])
#eval_metrics: ['AUC','ndcg']

# check performance 
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,precision_score, recall_score
pred_train=lgb_model.predict_proba(x_train)[:,1]
pred_test=lgb_model.predict_proba(x_val)[:,1]

print(roc_auc_score(y_train,pred_train))
print(average_precision_score(y_train,pred_train))


### feature importance ###
feat_imp = pd.Series(lgb_model.feature_importances_, index=x_train.columns)
feat_imp.sort_values(ascending=False).head(20)

# Gridsearch (optional)
from sklearn.model_selection import GridSearchCV
# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.02,0.05],
    'subsample': [0.8, 0.9]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='average_precision', cv=3, verbose=1)
# Perform grid search
grid_search.fit(x_train.values, y_train)
best_model=grid_search.best_estimator_ #show the best parameter of gridsearch
pred_test=best_model.predict_proba(x_val)[:,1]


# find the best threshold for f1 score
thresholds = np.linspace(0, 1, 50)

f_scores = []
precisions = []
recalls = []


for threshold in thresholds:
    f_scores.append(f1_score(y_train, pred_train > threshold))
    precisions.append(precision_score(y_train,pred_train > threshold))
    recalls.append(recall_score(y_train, pred_train > threshold))


selected_threshold=thresholds[np.argmax(f_scores)]
print(selected_threshold) # use the optimal threshold to be the cut point 

### precision, recall, f1 in training
print(precision_score(y_train, pred_train > selected_threshold))
print(recall_score(y_train, pred_train > selected_threshold))
print(f1_score(y_train, pred_train > selected_threshold))


### precision, recall, f1 in testing
print(precision_score(y_val, pred_test > selected_threshold))
print(recall_score(y_val, pred_test > selected_threshold))
print(f1_score(y_val, pred_test > selected_threshold))



##### MLP ####

X = df[features].copy()
y = df[label].copy()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# impute (更标准)
imputer = SimpleImputer(strategy="median")
x_train = imputer.fit_transform(x_train)
x_val  = imputer.transform(x_val)

print('x_train shape',x_train.shape)

#scaling
# 可选：如果你确认这些都是连续数值才 scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val  = scaler.transform(x_val)

# to torch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_val_tensor  = torch.tensor(x_val, dtype=torch.float32)
print('x_train_shape',x_train_tensor.shape)

y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1) #reshapes a tensor so it has exactly one column
print('y_train_shape',y_train.shape,y_train.to_numpy().shape, y_train_tensor.shape)
y_val_tensor  = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Compact MLP with additional dropout to limit overfitting on 36-dim inputs
class NN(nn.Module):
    def __init__(self, n_features):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)
    

### model setup
model = NN(x_train_tensor.shape[1])
criterion = torch.nn.BCEWithLogitsLoss()

'''
Weight decay is L2 regularization built into the optimizer: every update shrinks the parameters slightly toward zero, discouraging large weights.
'''
optimizer = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)


from sklearn.metrics import roc_auc_score

epochs = 20
best_val_auc = float("-inf")
best_state = None

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0


    for data, target in train_loader:
        #Think “Clear → Predict → Compare → Backprop → Update.”
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() # update weights
        ### loss.item(), converts the scalar loss tensor for the current batch into a plain Python float (e.g., 0.1234)
        epoch_loss += loss.item() * len(data)

    avg_train_loss = epoch_loss / len(train_loader.dataset)

    model.eval() # eval mode

    with torch.no_grad():
        train_logits = model(x_train_tensor)
        train_auc = roc_auc_score(y_train_tensor, train_logits)

    with torch.no_grad():
        val_logits = model(x_val_tensor)
        val_auc = roc_auc_score(y_val_tensor, val_logits)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")


with torch.no_grad():
    y_pred = model(x_train_tensor) # output shape [rows,1 ]
    auc = roc_auc_score(y_train_tensor, y_pred)
    print("Train AUC:", auc)

with torch.no_grad():
    y_pred = model(x_val_tensor)
    auc = roc_auc_score(y_val_tensor, y_pred)
    print("Val AUC:", auc)