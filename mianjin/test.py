
#from sklearn.datasets import make_classification


# unsupervised learning
# class knn , initilize the number of cluster, max_iter, tolerance
# methods, fit (x), assign_labels

import numpy.random as nd
import numpy as np

class Kmeans:
    def __init__(self,n_clusters=3, max_iter=100, tolerance=1e-4):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.tolerance=tolerance

    def fit(self,x):
        # random select centroid
        self.centroids=x[np.random.choice(x.shape[0],size=self.n_clusters)]
        
        for _ in range(self.max_iter):
            new_label=self.assign_labels(x) #lables is a list of x.shape[0]
            # calculate the new centroids
            new_centroids=[]
            for k in range(self.n_clusters):
                new_centroids.append(x[new_label==k].mean(axis=0))
            
            # check the new_centroid and the existing centroid
            if np.allclose(new_centroids,self.centroids,rtol=self.tolerance):
                break
            
            self.centroids=new_centroids

        #退出循环后，最后定义labels
        self.labels=self.assign_labels(x)

    def assign_labels(self, x):
        # compare the distance between x and each centroids
        dist=[]
        for i in range(self.n_clusters):
            c_dist=np.sum((x-self.centroids[i])**2,axis=1)
            dist.append(c_dist)

        labels=np.argmin(dist,axis=0)
        return labels

#---- test case -------

x=nd.random(size=(100,2))

model=Kmeans()

model.fit(x)
#print(model.labels)
            





