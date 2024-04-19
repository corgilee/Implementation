
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self,lr,iteration):
        self.lr=lr
        self.iter=iteration

    def sigmoid(self,x):
        z=np.dot(x,self.w)+self.b
        return 1/(1+np.exp(-z))

    def fit(self,x,y):
        m,n=x.shape
        self.w=np.random.random(size=n)
        self.b=np.random.random(size=1)


        for _ in range(self.iter):
            pred=self.sigmoid(x)

            gap=pred-y
            #calculate the gradient
            dw=1/m*np.dot(x.T,gap)
            db=1/m*np.sum(gap)

            self.w-=lr*dw
            break


# Generate a more complex dataset
x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=15, n_redundant=5, random_state=42)

# Splitting the dataset into training and testing sets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(x.shape[0] * split_ratio)

x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model=LogisticRegression(lr=0.01, iteration=100)
model.fit(x_train, y_train)




