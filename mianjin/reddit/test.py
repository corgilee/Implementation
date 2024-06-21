import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Synthetic data generation
def generate_data(num_samples):
    np.random.seed(42)
    num_features = 5
    max_history_length = 10
    num_merchants = 50

    # Numerical features
    numerical_features = np.random.randn(num_samples, num_features)

    # Shopping history (sequence of merchant IDs)
    shopping_history = np.random.randint(0, num_merchants, (num_samples, max_history_length))

    # Binary labels
    labels = np.random.randint(0, 2, num_samples)

    return numerical_features, shopping_history, labels

# Custom dataset
class ShoppingDataset(Dataset):
    def __init__(self, numerical_features, shopping_history, labels):
        self.numerical_features = numerical_features
        self.shopping_history = shopping_history
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numerical_features': torch.tensor(self.numerical_features[idx], dtype=torch.float),
            'shopping_history': torch.tensor(self.shopping_history[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)  # Use float for binary cross-entropy loss
        }

# Transformer model for shopping history
class ShoppingTransformer(nn.Module):
    def __init__(self, num_merchants, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(ShoppingTransformer, self).__init__()
        self.embedding = nn.Embedding(num_merchants, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        return output

# MLP classifier with Transformer for shopping history
class MLPClassifier(nn.Module):
    def __init__(self, num_features, num_merchants, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(MLPClassifier, self).__init__()
        self.shopping_transformer = ShoppingTransformer(num_merchants, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(num_features + d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single output for binary classification
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, numerical_features, shopping_history):
        transformer_output = self.shopping_transformer(shopping_history).mean(dim=1)
        combined_features = torch.cat((numerical_features, transformer_output), dim=1)
        output = self.mlp(combined_features)
        return output

# Data preparation
numerical_features, shopping_history, labels = generate_data(1000)
X_train_num, X_test_num, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
    numerical_features, shopping_history, labels, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# Create datasets and dataloaders
train_dataset = ShoppingDataset(X_train_num, X_train_hist, y_train)
test_dataset = ShoppingDataset(X_test_num, X_test_hist, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model instantiation
num_features = X_train_num.shape[1]
num_merchants = np.max(shopping_history) + 1
d_model = 32
nhead = 4
num_encoder_layers = 2
dim_feedforward = 128
dropout = 0.1

model = MLPClassifier(num_features, num_merchants, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        numerical_features = batch['numerical_features']
        shopping_history = batch['shopping_history']
        labels = batch['labels']

        outputs = model(numerical_features, shopping_history).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        numerical_features = batch['numerical_features']
        shopping_history = batch['shopping_history']
        labels = batch['labels']

        outputs = model(numerical_features, shopping_history).squeeze()
        preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy}')
