import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score


# class of logistic regression
# initialize learning rate, iteration
# define sigmoid
# define fit, 
# define predict_proba

class LogisticRegression:
    def __init__(self, learning_rate, iteration):
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.weights=None # 初始化为none
        self.bias=None

    def sigmoid(self,z):
        out=1/(1+np.exp(-z))

        return out

    def fit(self, x, y):
        m,n=x.shape
        #initiate weight
        self.weights=np.random.random(n) # =number of features
        self.bias=0

        for _ in range(self.iteration):
            z=np.dot(x, self.weights)+self.bias
            pred=self.sigmoid(z)

            # Gradient calculations
            # chain rule 推导（向量形式）：
            # 1) z = Xw + b
            # 2) a = sigmoid(z)
            # 3) L = -(1/m) * sum(y*log(a) + (1-y)*log(1-a))
            # chain rule 展开：
            # dL/dw = dL/da * da/dz * dz/dw
            # dL/db = dL/da * da/dz * dz/db

            # dL/da = -(1/m) * (y/a - (1-y)/(1-a))
            # da/dz = a*(1-a)
            # => dL/dz = (1/m) * (a - y)   (化简后)

            # dz/dw = X, dz/db = 1
            # => dL/dw = (1/m) * X^T (a - y)
            # => dL/db = (1/m) * sum(a - y)

            dw=(1/m)*np.dot(x.T,(pred-y)) # 注意这里是x.T
            db=(1/m)*np.sum(pred-y)

            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict_proba(self, x):
        z=np.dot(x,self.weights)+self.bias
        return self.sigmoid(z)


    # def predict(self, X, threshold=0.5):
    #     probabilities = self.predict_proba(X)
    #     return np.where(probabilities >= threshold, 1, 0)

'''
# Assuming X_train, y_train are defined and do not include the bias term in X_train
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train[:,1:], y_train)  # We exclude the bias column if previously added

# Now you can access the separated weights and bias
print("Weights:", model.weights)
print("Bias:", model.bias)

# To make predictions:
# predictions = model.predict(X_test[:,1:])
'''


# Generate a more complex dataset
x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=15, n_redundant=5, random_state=42)

# Splitting the dataset into training and testing sets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(x.shape[0] * split_ratio)

x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model=LogisticRegression(learning_rate=0.01, iteration=100)
model.fit(x_train, y_train)
pred=model.predict_proba(x_test)

print('roc auc score is',roc_auc_score(y_test, pred))
print('The weights are', model.weights)
print('The bias is',model.bias)
