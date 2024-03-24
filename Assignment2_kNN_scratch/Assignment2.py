# import packages
import sklearn.datasets 
#from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

X,y = sklearn.datasets.make_classification(n_samples=10,n_features=4)

#plt.style.use('seaborn-v0_8')
#plt.scatter(X[:,0], X[:,1], c=y, marker= '*',s=100,edgecolors='black')
#plt.show()

data = pd.DataFrame(X)
data.to_excel('sample_data.xlsx', sheet_name='sheet1', index=False)
print("0:",data.iloc[0])
print(data)

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    print("----- To test:\n", test_row)
    distances = pd.DataFrame(columns = ['Class', 'Distance'])
    i = 0
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        conc = pd.DataFrame({'Class' : y[i], 'Distance' : dist}, index=[i])
        distances = pd.concat([distances, conc],ignore_index = False)
        i = i+1
    distances.sort_values(by=['Distance'], inplace=True)
    print("Distances:\n", distances,"\n")
    distances = distances[distances.Distance > 0]
    neighbors = distances.head(3)
    print("neighbors\n",neighbors,"\n")
    prediction = neighbors['Class'].value_counts().idxmax()
    print("prediction\n",prediction,"\n")    
    return neighbors

neighbors = get_neighbors(X, X[3], 3)