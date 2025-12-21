
#---plot --- 
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Category', hue='Target', data=df)
plt.title('Count Plot of Categorical Feature by Binary Target')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.legend(title='Target', loc='upper right')
plt.show()

#------tfidf ------------
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

# Convert the TF-IDF matrix to a dense array for easier manipulation (optional)
tfidf_matrix_dense = tfidf_matrix.toarray()

# Get the feature names (words) from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()


##----- MLP --------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

x_train, x_val = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_val, dtype=torch.float32)
y_train, y_val = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features + n_categories, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Output layer for binary classification with a single output neuron
            nn.Sigmoid()       # Sigmoid activation function for the final output
        )

    def forward(self, x):
        return self.layers(x)


model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

def train_model(num_epochs, model, loaders):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in loaders['train']:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Train the model
train_model(10, model, {'train': train_loader})



# increas the positive ratio to ~x% in x_train,if needed
# ---- combine X + y for easy filtering ----
train_df = x_train.copy()
train_df["label"] = y_train.values

# ---- split positives / negatives ----
df_pos = train_df[train_df["label"] == 1]
df_neg = train_df[train_df["label"] == 0]

P = len(df_pos)
N_neg_keep = P*30

# ---- downsample negatives ----
df_neg_down = df_neg.sample(n=N_neg_keep,replace=False)

# ---- recombine & shuffle ----
train_balanced = (
    pd.concat([df_pos, df_neg_down], axis=0)
      .sample(frac=1.0, random_state=42)
      .reset_index(drop=True)
)

# ---- split back to X / y ----
x_train_bal = train_balanced.drop(columns=["label"])
y_train_bal = train_balanced["label"]

print("Train positive rate:",
      y_train_bal.mean() * 100)

x_train=x_train_bal.copy()
y_train=y_train_bal.copy()