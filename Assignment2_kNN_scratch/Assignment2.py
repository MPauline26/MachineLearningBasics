# import packages
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import pandas as pd
from math import sqrt
import numpy as np

# calculate distance of base row to other rows
# compute sum of distances for each column
# return sqrt because euclidean distance
def kNN_distance(row_base, row_oth):
    distance = 0.0
    for i in range(len(row_base)): 
        distance += (row_base[i] - row_oth[i])**2
    return sqrt(distance) 

# call kNN_distance() for each row and save into DataFrame distance_all
# sort resulting DataFrame by distance
# take only k-nearest neighbors
# predict Class by taking most occuring class among k-nearest neighbors in train-data
def kNN(train_X_data, train_y_data, test_row, k_num):

    i = 0
    distance_all = pd.DataFrame(columns = ['Class', 'Distance'])

    for train_row in train_X_data:
        distance = kNN_distance(test_row, train_row)
        conc = pd.DataFrame({'Class' : train_y_data[i], 'Distance' : distance}, index=[i])
        distance_all = pd.concat([distance_all, conc], ignore_index = False)
        i += 1

    distance_all.sort_values(by=['Distance'], inplace=True)

    neighbors = distance_all.head(k_num)
    prediction = neighbors['Class'].value_counts().idxmax()

    return prediction

# calls kNN() for each row in test-data and makes class prediction
# compare predicted value to actual value in test data
# calculate accuracy ratio: proportion of correctly predicted occurences to total number
def accuracy(train_X_data, train_y_data, test_X_data, test_y_data, k_num = 3):

    j = 0
    accuracy_list = pd.DataFrame(columns = ['Actual', 'Predicted'])

    for row in test_X_data:
        predicted_value = kNN(train_X_data, train_y_data, test_X_data[j], k_num)
        conc = pd.DataFrame({'Actual' : test_y_data[j], 'Predicted' : predicted_value}, index=[j])
        accuracy_list = pd.concat([accuracy_list, conc],ignore_index = False)
        j += 1

    accuracy_list['Correctly_Predicted'] = np.where(accuracy_list['Actual'] == accuracy_list['Predicted'], True, False)
    accuracy_ratio = accuracy_list["Correctly_Predicted"].mean()

    return accuracy_ratio

# set constants
KNEIGHBOR_NUM = 5

SEED_DATASET = 20220919
ROWS_NUM = 100
FEATURES_NUM = 4

SEED_KFOLD = 20220919
KFOLD_NUM = 10

SCALING = False

# create dataset
X,y = make_classification(n_samples=ROWS_NUM,n_features=FEATURES_NUM,random_state=SEED_DATASET)

if SCALING == True :
    print("Mean of each column: ", np.mean(X, axis=0))
    print("St. dev of each column: ",np.std(X, axis=0), "\n")
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0)

# call k-fold cross validation
kf = KFold(n_splits=KFOLD_NUM, shuffle=True, random_state=SEED_KFOLD)

# Calculate accuracy ratio for each evaluation, save in accuracy_ratio_sample
# Compute average for final performance estimate
accuracy_ratio_sample = pd.DataFrame(columns = ['Train', 'Test','Accuracy_Ratio'])
m = 0

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    accuracy_ratio = accuracy(X_train, y_train, X_test, y_test, KNEIGHBOR_NUM)
    conc = pd.DataFrame({'Train' : [train_index], 'Test' : [test_index], 'Accuracy_Ratio' : accuracy_ratio}, index=[m])
    accuracy_ratio_sample = pd.concat([accuracy_ratio_sample, conc],ignore_index = False)
    m += 1

accuracy_ratio = accuracy_ratio_sample["Accuracy_Ratio"].mean()
print(accuracy_ratio_sample)

print("Final performance estimate: ", accuracy_ratio)

# Sample size: 4 Features, 100 observations
# 10 fold cross validation

# using k = 3, no feature scaling - final performance estimate: 0.91
# using k = 5, no feature scaling - final performance estimate: 0.92
# using k = 7, no feature scaling - final performance estimate: 0.89

# using k = 3, standardized feature scaling - final performance estimate: 0.93
# using k = 5, standardized feature scaling - final performance estimate: 0.93
# using k = 7, standardized feature scaling - final performance estimate: 0.91