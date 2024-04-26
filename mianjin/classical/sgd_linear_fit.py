'''
stochastic gradient descent
assuming function is: y=2*x+1

initialize w, b, learning_rate, epoches

for each epoch, for each data point
pred=w*x+b
loss_function=(pred-y)**2
gradient_w=2*(pred-y)*x
gradient_b=2*(pred-y) , 注意这里是pred-y, 不是y-pred
update the gradient


'''


import numpy as np
x=np.arange(6)
y=2*x+1+np.random.uniform(-1, 1)

print(x)
print(y)

w=0
b=0
lr=0.01
epoches=10

for epoch in range(epoches):
    for i in range(len(x)):
        xi=x[i]
        yi=y[i]
        yi_pred=w*xi+b
        loss=(yi_pred-yi)**2
        #compute gradient
        dw=2*(yi_pred-yi)*xi
        db=2*(yi_pred-yi)

        #update parameters
        w-=dw*lr
        b-=db*lr
print(f'After training, the weight w is {np.round(w,2)}, the bias b is {np.round(b,2)} ')


'''
import numpy as np

# Example data: y = 2*x + 1
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Parameters initialization
w = 0.0  # Weight
b = 0.0  # Bias
learning_rate = 0.01
n_epochs = 1000

# Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        yi_pred = w * xi + b  # Prediction
        loss = (yi_pred - yi)**2

        # Compute gradients
        dw = 2 * (yi_pred - yi) * xi
        db = 2 * (yi_pred - yi)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

print(f"Trained weights: w = {w:.2f}, b = {b:.2f}")
'''