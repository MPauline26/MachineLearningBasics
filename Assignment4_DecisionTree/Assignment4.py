# to surpress FutureWarning message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import packages
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# set constants
ROWS_NUM = 500 # size of sample (train and test)
FEATURES_NUM = 5 # number of features, also needs to be adapted in row 20

# create dataset
X,y = make_classification(n_samples=ROWS_NUM,n_features=FEATURES_NUM, class_sep=2)
data_X = np.array(X)
data_y = np.array(y)
data = pd.DataFrame(np.append(data_X ,data_y.reshape(-1,1),axis=1), columns=['VAR1','VAR2','VAR3','VAR4','VAR5', 'target'])
data.target = data.target.astype('int')

# splitting data into test and train
train, test = train_test_split(data, test_size=0.2)

def gini_calculation(data_input, var, value, target):

    # create temporary variable to split variable into two buckets
    data_input['temp_var'] = np.where(data_input[var]<value, 'Bucket1', 'Bucket2')
    
    # count frequency of target in each bucket
    count = data_input.groupby([pd.Categorical(data_input.target), 'temp_var']).size().fillna(0)
    count = np.array(count.values.tolist())
    count = np.resize(count, (2, 2))

    # count total frequency of each bucket
    total = data_input.value_counts(["temp_var"]).sort_index()
    total = total.to_numpy()
    total = np.resize(total, (1, 2))

    # calculate gini of each bucket
    gini_indiv = (1 - np.sum((count/total)**2, axis = 0))

    # calculate weights and total gini
    weights = np.sum(count, axis=0)/len(data_input.index)
    gini = np.sum(gini_indiv*weights)

    # drop temporary variable
    data_input.drop(['temp_var'], axis=1, inplace=True)

    return gini

# create list of possible splits (all unique values of all features)
gini_list = pd.DataFrame(columns = ['variable','unique_values','gini'])

for column in train:
    if column == 'target':
        continue

    # create temporary DataFrame to append list
    temp = pd.DataFrame()
    temp['unique_values'] = pd.DataFrame(set(train[column]))
    temp['variable'] = column
    gini_list = gini_list.append(temp, ignore_index=True)

# calculate ginis for all splits
for index, row in gini_list.iterrows():
    gini_list.iloc[index, 2] = gini_calculation(train, row['variable'], row['unique_values'], 'target')

# sort list by ascending gini
gini_list = gini_list.sort_values(by=['gini']).reset_index(drop=True)

# determine most occuring class per split
class1 = train[train[gini_list.iloc[0,0]]<gini_list.iloc[0,1]].target.mode()
class2 = train[train[gini_list.iloc[0,0]]>=gini_list.iloc[0,1]].target.mode()

# variable to split: gini_list.iloc[0,0]
# value to split by: gini_list.iloc[0,1]
# class in first branch (<  than x): class1
# class in first branch (>= than x): class2

# apply split on test sample, make predictions and calculate accuracy 
test['prediction'] = np.where(test[gini_list.iloc[0,0]]<gini_list.iloc[0,1], class1, class2)
test['Correctly_Predicted'] = np.where(test['target'] == test['prediction'], True, False)
accuracy_ratio = test["Correctly_Predicted"].mean()

# print results
print("DT is split by ", gini_list.iloc[0,0], " at ", gini_list.iloc[0,1])
print("First branch is: ", class1.to_string(index=False))
print("Second branch is: ", class2.to_string(index=False))
print("Gini is: ", gini_list.iloc[0,2])
print("Accuracy ratio on test sample is: ", accuracy_ratio)
if not test[test['Correctly_Predicted']==False].empty:
    print("Incorrectly categorized: \n", test[test['Correctly_Predicted']==False].head())