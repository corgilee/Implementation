#source .venv/bin/activate

### Data Loading and Initial Exploration

import numpy as np 
import pandas as pd 
import csv 
df = pd.read_csv("/Users/wenjiezhao/Documents/creditcard_fraud.csv") 
df.shape


# Get a summary of the DataFrame including data types, non-null values, and memory usage
df.info()
#data.columns.tolist() 
label="Class"

print("--- Target Variable Distribution ('has_product_click') ---") 
click_counts = df[label].value_counts() 
print(click_counts) 
print(f"pos rate: {click_counts[1] / len(df) * 100 : .2f}%") 


#Feature Engineering: Encoding and Missing Value Imputation

## numercial features
num_cols=df.select_dtypes(include=['float64','int64']).columns.tolist()

# drop necessary columns


#id_cols_to_drop = ['has_product_click', 'request_id_anon', 'retailer_token_anon'] 
id_cols_to_drop = ['Class'] 
num_cols = [col for col in num_cols if col not in id_cols_to_drop] 

print('len of numerical colums', len(num_cols))

## categorical featrues

cat_cols=df.select_dtypes(include=['object']).columns.tolist()
print(cat_cols)
# check the unique values to judge if string or limited category
print(df[cat_cols].nunique())

#looks like filter_string can be treated as categorical data in the later steps
cat_cols=["filter_string"]
cat_cols=[]


## quickly check the missing rate first and than do the split before imputation
missing_cols=(df.isna().sum()*100/len(df)).reset_index(name="missing_rate").query("missing_rate>0").sort_values(by="missing_rate",ascending=False)
#print(missing_cols)

### optional 
# columns with < 80% missing
cols_keep = missing_cols.loc[
    missing_cols["missing_rate"] < 80, "column"
].tolist()

# keep also columns with 0% missing
cols_keep += [
    c for c in df.columns
    if c not in missing_cols["column"].tolist()
]
#####

# optino 2: target encoding
query_mean=pd.DataFrame(df.groupby(['query_text'])[label].mean()).reset_index()
query_mean_dict=dict(zip(query_mean['query_text'],query_mean[label]))
df['query_text_encoding']=df['query_text'].map(query_mean_dict)


### split ###
from sklearn.model_selection import train_test_split

X=df[num_cols+cat_cols]
y=df[label]

x_train, x_val, y_train, y_val = train_test_split(
X, y, test_size=0.1, random_state=42,stratify=y)

print(x_train.shape)
print(x_val.shape)



### impute numercial data with missing_indicator and median first ###
for col in num_cols:
    x_train[col+"_missing"]=x_train[col].isna().astype(int)
    x_val[col+"_missing"]=x_val[col].isna().astype(int)

num_cols_train_median=x_train[num_cols].median()
#print(num_cols_train_median)
### fill na
x_train[num_cols]=x_train[num_cols].fillna(num_cols_train_median)
x_val[num_cols]=x_val[num_cols].fillna(num_cols_train_median)

print('x_train shape', x_train.shape)
print('x_val shape',x_val.shape)

### categorical imputation ###
# explicit missing token, then generate dummies for selected catogrical features
# --- categorical: explicit missing token ---
x_train[cat_cols] = x_train[cat_cols].fillna("__MISSING__")
x_val[cat_cols] = x_val[cat_cols].fillna("__MISSING__")

x_train= pd.get_dummies(x_train, columns=cat_cols, drop_first=False)
x_val= pd.get_dummies(x_val, columns=cat_cols, drop_first=False)

#I reindex validation features to match the training feature space.
# This ensures the model sees the exact same columns in train and validation; unseen categories are dropped, 
# and missing ones are filled with zeros.
x_val = x_val.reindex(columns=x_train.columns, fill_value=0)



# clean the column name to avoid potential error, r"[ ... ]"
def clean_col_names(df):
    df.columns = df.columns.str.replace(
        r"[\[\]\:\|\"]",
        "_",
        regex=True
    )
    return df

x_train=clean_col_names(x_train)
x_val=clean_col_names(x_val)



### textual preprocessing ##
## if lower case of the string constains a certain word like "exam"
'''
df["text_has_exam"] = (
    df["text_column"].str.lower().str.contains("exam", na=False).astype(int)
)

## if col 1 (string format) are included in col 2 (string format)
df["col1_in_col2"] = (
    df.apply(lambda row: str(row["col1"]).lower() in str(row["col2"]).lower(),
             axis=1)
      .astype(int)
)
'''

### Xgboost for classification ###
import xgboost as xgb
dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)

# keep the original x_train/y_train split; no manual downsampling
pos = y_train.sum()
neg = len(y_train) - pos
scale = neg / max(pos, 1)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['aucpr'],
    'eta': 0.03, # learning_rate
    'max_depth': 3,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 10,
    'scale_pos_weight': scale,
    'tree_method': 'hist',
    'seed': 42,
}


dtrain_eval = xgb.DMatrix(x_train)
dval_eval = xgb.DMatrix(x_val)

watchlist = [(dtrain, 'train'), (dval, 'eval')]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=25,
)

train_pred=model.predict(dtrain_eval)
val_pred=model.predict(dval_eval)

from sklearn.metrics import roc_auc_score, average_precision_score,precision_score,recall_score,f1_score
#print(roc_auc_score(y_train, train_pred))
#print(roc_auc_score(y_val, val_pred))


print('train prauc', average_precision_score(y_train, train_pred))
print('val prauc',average_precision_score(y_val, val_pred))

# Feature Importance
importance = model.get_score(importance_type='gain') 
feature_importances = pd.Series(importance).sort_values(ascending=False) 
print("--- Top 5 Important Features (by Gain) ---") 
print(feature_importances.head(5))


# find the best threshold for f1 score
from sklearn.metrics import f1_score, precision_score, recall_score


thresholds = np.linspace(0, 0.9, 10)

f_scores = []
precisions = []
recalls = []


for threshold in thresholds:
    print('threshold:', threshold)
    f_scores.append(f1_score(y_train, train_pred > threshold))
    precisions.append(precision_score(y_train,train_pred > threshold))
    recalls.append(recall_score(y_train, train_pred > threshold))


selected_threshold=thresholds[np.argmax(f_scores)]
print('selected_threshold',selected_threshold) # use the optimal threshold to be the cut point 

## precision, recall, f1 in training
print('train recall',recall_score(y_train, train_pred > selected_threshold))
print('train precision', precision_score(y_train, train_pred > selected_threshold))
print('train f1',f1_score(y_train, train_pred > selected_threshold))


### precision, recall, f1 in testing
print('val recall', recall_score(y_val, val_pred > selected_threshold))
print('val precision ', precision_score(y_val, val_pred > selected_threshold))

print('val f1', f1_score(y_val, val_pred > selected_threshold))

### optinal calibration

from sklearn.isotonic import IsotonicRegression

# calibrate raw model scores so their average probability matches the true positive rate
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(val_pred, y_val)

train_calibrated = calibrator.transform(train_pred)
val_calibrated = calibrator.transform(val_pred)

print('calibrated train pos rate', train_calibrated.mean())
print('true train pos rate', y_train.mean())
print('calibrated val pos rate', val_calibrated.mean())
print('true val pos rate', y_val.mean())



##### Neural network  ######
'''


print("------------------------------")
import torch

# featrues
features = num_cols

X = df[features].copy()
y = df[label].copy()

from sklearn.model_selection import train_test_split
X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# impute (更标准)
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
x_val  = imputer.transform(x_val)

print('x_train shape',X_train.shape)

#scaling
# 可选：如果你确认这些都是连续数值才 scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_val  = scaler.transform(x_val)

# to torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
x_val_tensor  = torch.tensor(x_val, dtype=torch.float32)
print('x_train_shape',X_train_tensor.shape)

y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1) #reshapes a tensor so it has exactly one column
print('y_train_shape',y_train.shape,y_train.to_numpy().shape, y_train_tensor.shape)
y_val_tensor  = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Compact MLP with additional dropout to limit overfitting on 36-dim inputs
import torch.nn as nn

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
model = NN(X_train_tensor.shape[1])


pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


#Weight decay is L2 regularization built into the optimizer: every update shrinks the parameters slightly toward zero, discouraging large weights.

import torch.optim as optim
optimizer = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)


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
        train_logits = model(X_train_tensor)
        train_auc = roc_auc_score(y_train_tensor, train_logits)

    with torch.no_grad():
        val_logits = model(x_val_tensor)
        val_auc = roc_auc_score(y_val_tensor, val_logits)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
'''
