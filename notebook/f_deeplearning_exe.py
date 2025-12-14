import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# url = "https://drive.google.com/file/d/1iqEDDljacKQP5UmlWKzW9uX0hz4AWkDQ/view?usp=sharing"
# gdown.download(url, output="faire-ml-rank-small.csv",fuzzy=True)

data = pd.read_csv("faire-ml-rank-small.csv")

data.shape

print('data.info')
#print(data.info())

num_features=data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#print('num_features: ', num_features)

label = 'has_product_click'
# postiive rate
click_counts=data[label].value_counts()
print('pos rate', click_counts[1]/len(data))


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




features = [x for x in data.columns if x.startswith('retailerbrand') or x.startswith('product')]
features=features+['position','page_number']
#print('features', features)


label = 'has_product_click'

X = data[features].copy()
y = data[label].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

from sklearn.impute import SimpleImputer

# impute (更标准)
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

print('x_train shape',X_train.shape)

# 可选：如果你确认这些都是连续数值才 scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# to torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1) #reshapes a tensor so it has exactly one column
print('y_train_shape',y_train.shape,y_train.to_numpy().shape, y_train_tensor.shape)
y_test_tensor  = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)


from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
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
    

model = NN(X_train_tensor.shape[1])
criterion = torch.nn.BCEWithLogitsLoss()

'''
Weight decay is L2 regularization built into the optimizer: every update shrinks the parameters slightly toward zero, discouraging large weights.
'''
optimizer = optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)

epochs = 20
best_val_auc = float("-inf")
best_state = None

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        ### loss.item(), converts the scalar loss tensor for the current batch into a plain Python float (e.g., 0.1234)
        epoch_loss += loss.item() * len(data)

    avg_train_loss = epoch_loss / len(train_loader.dataset)

    model.eval()
    
    with torch.no_grad():
        train_logits = model(X_train_tensor)
        train_auc = roc_auc_score(y_train_tensor, train_logits)

    with torch.no_grad():
        val_logits = model(X_test_tensor)
        val_auc = roc_auc_score(y_test_tensor, val_logits)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    #### early stop #####
    # if val_auc > best_val_auc:
    #     best_val_auc = val_auc
    #     best_state = model.state_dict()
    # else:
    #     print("Validation AUC did not improve; stopping early.")
    #     if best_state is not None:
    #         model.load_state_dict(best_state)
    #     break


with torch.no_grad():
    y_pred = model(X_train_tensor) # output shape [rows,1 ]
    auc = roc_auc_score(y_train_tensor, y_pred)
    print("Train AUC:", auc)

with torch.no_grad():
    y_pred = model(X_test_tensor)
    auc = roc_auc_score(y_test_tensor, y_pred)
    print("Test AUC:", auc)
