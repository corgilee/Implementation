from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

x,y = load_diabetes(return_X_y=True)


# gbm is composed of many decision trees

class CustomGradientBoostingRegressor:
    def __init__(self, learning_rate, n_estimators, max_depth=1):
        self.learning_rate=learning_rate
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.trees=[]

    def fit(self,x,y):
        #initialize F0
        self.F0=y.mean() 
        Fm=self.F0

        for _ in range(self.n_estimators):
            r=y-Fm # residual
            tree=DecisionTreeRegressor(max_depth=self.max_depth,random_state=0)
            tree.fit(x,r)
            self.trees.append(tree)
            gamma=tree.predict(x)
            Fm+=self.learning_rate*gamma

    def predict(self, x):
        Fm=self.F0
        for i in range(self.n_estimators):
            Fm+=self.learning_rate*self.trees[i].predict(x)

        return Fm

gbm=CustomGradientBoostingRegressor(learning_rate=0.1,n_estimators=50,max_depth=5)

gbm.fit(x,y)
pred=gbm.predict(x)

#print(pred)

print(mean_squared_error(y,pred))

