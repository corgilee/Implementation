# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/

file="C:/Users/c/Documents/Python Scripts/Fraud/Data/knn_test.txt"

''''
Height, Weight, Age, Class
1.70, 65, 20, Programmer
1.90, 85, 33, Builder
1.78, 76, 31, Builder
1.73, 74, 24, Programmer
1.81, 75, 35, Builder
1.73, 70, 75, Scientist
1.80, 71, 63, Scientist
1.75, 69, 25, Programmer
'''''


f=open(file,"r")
lines=f.read().splitlines()
f.close()


#the first line is head
# get all the feature names
features=lines[0].split(',')[:-1] 
items=[]

for i in range(1,len(lines)):
    #transverse every lines
    line=lines[i].split(',')
    itemFeatures={"class": line[-1]}
    
    # Iterate through the features
    for j in range(len(features)):
        
        #Get the feature at index j
        f=features[j]
        
        #Convert feature value to float
        v=float(line[j])
        
        # Add feature value to dict
        itemFeatures[f]=v
    items.append(itemFeatures)
 
''''   
items:
    
[{'class': ' Programmer', 'Height': 1.7, ' Weight': 65.0, ' Age': 20.0},
 {'class': ' Builder', 'Height': 1.9, ' Weight': 85.0, ' Age': 33.0},
 {'class': ' Builder', 'Height': 1.78, ' Weight': 76.0, ' Age': 31.0},
 {'class': ' Programmer', 'Height': 1.73, ' Weight': 74.0, ' Age': 24.0},
 {'class': ' Builder', 'Height': 1.81, ' Weight': 75.0, ' Age': 35.0},
 {'class': ' Scientist', 'Height': 1.73, ' Weight': 70.0, ' Age': 75.0},
 {'class': ' Scientist', 'Height': 1.8, ' Weight': 71.0, ' Age': 63.0},
 {'class': ' Programmer', 'Height': 1.75, ' Weight': 69.0, ' Age': 25.0}]
'''''
### Auxiliary Function ##### def EuclideanDistance(x,y):
import math
def distance(x,y,features):
    S=0
    for feature in features:
        S+=math.pow(x[feature]-y[feature],2)
    return math.sqrt(S)

# use stack to save [distance, "label"]
def UpdateNeighbors(neighbors, item, distance,k):
    if len(neighbors)<k:
        neighbors.append([distance,item['class']])
        neighbors.sort()
    else:
        if neighbors[-1][0]>distance:
            neighbors[-1]=[distance,item['class']]
            neighbors.sort()
    return neighbors

def count_neighbor_class(neighbors,k):
    count={}
    for i in range(k):
        if neighbors[i][1] not in count:
            count[neighbors[i][1]]=1
        else:
            count[neighbors[i][1]]+=1
    return count

def max_count(count):
    maximum=-1
    label=" "
    for key in count.keys():
        if count[key]>maximum:
            maximum=count[key]
            label=key
    return label


### core function ####
def knn(items,features,k, new):
    neighbors=[]
    for item in items:
        dist=distance(item,new, features)
        
        neighbors=UpdateNeighbors(neighbors, item, dist,k)
    count=count_neighbor_class(neighbors,k)
    label=max_count(count)
    return label

  
new={'Height': 1.3, ' Weight': 68.0, ' Age': 17.0}

if __name__ == '__main__':
    
    print(knn(items,features,3,new))


    


        



    
    
        