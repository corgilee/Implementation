
import numpy as np


'''
class kMeans:
    init: n_cluster, max_round, tol
    fit
    assign_labels
'''
class KMeans:
    def __init__(self,n_clusters, max_round=100,tol=1e-4):
        self.n_cluster=n_clusters
        self.max_round=max_round
        self.tol=tol

    def fit(self,x):
        # 先random 产生 中心
        self.centroids=x[np.random.choice(x.shape[0],size=self.n_cluster)] # 选了index之后一定要套上x

        #然后进入循环
        for i in range(self.max_round):
            #先把每个x 都assign 上label
            self.labels=self.assign_labels(x)
            #print(len(self.labels))

            #然后算一下new centroids

            new_centroids=[x[self.labels==k].mean(axis=0) for k in range(self.n_cluster)]

            #print(len(new_centroids))

            if np.allclose(new_centroids,self.centroids,rtol=self.tol):
                break
            
            self.centroids=new_centroids
        #最后assign label
        self.label=self.assign_labels(x)

    def assign_labels(self,xs):
        # 这个xs 是好多个x 组成的
        # 需要有self.centroids
        dists=[]
        for i in range(self.n_cluster):
            dist=np.sqrt(np.sum((xs-self.centroids[i])**2,axis=1))
            #print(dist)
            dists.append(dist)
        #print(dists)
        index=np.argmin(dists,axis=0)
        
        return index

np.random.seed(0)
x = np.random.randn(100, 2) # 100 points in 2D

# Instantiate and fit KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
print(kmeans.label)


# kmeans = KMeans(n_clusters=3)
# kmeans.fit(x)
# print(kmeans.label)