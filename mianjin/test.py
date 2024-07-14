import numpy as np

p=[0.1,0.2,0.3,0.4]
print(p)

p[-1] = 1 - np.sum(p[0:-1])

x=[1,2,3,4]
while True:
    np.random.choice(x, p = p)
    break

