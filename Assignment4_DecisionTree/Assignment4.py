# to surpress FutureWarning message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import packages
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from math import sqrt
import pandas as pd
import numpy as np

SEED_DATASET = 20220919
ROWS_NUM = 100
FEATURES_NUM = 4

# create dataset
X,y = make_classification(n_samples=ROWS_NUM,n_features=FEATURES_NUM,random_state=SEED_DATASET, class_sep=2)

data_X = np.array(X)
data_y = np.array(y)
data = pd.DataFrame(np.append(data_X ,data_y.reshape(-1,1),axis=1), columns=['test1','test2','test3','test4','target'])
data.target = data.target.astype('int')
print(data.head())

def gini_calculation(data_input, var, value,target):

    data['temp_var'] = np.where(data[var]<value, 'Bucket1', 'Bucket2')
    
    count = data_input.groupby([pd.Categorical(data_input.target), 'temp_var']).size().fillna(0)
    count = np.array(count.values.tolist())
    count = np.resize(count, (2, 2))

    total = data_input.value_counts(["temp_var"]).sort_index()
    total = total.to_numpy()
    total = np.resize(total, (1,2))

    gini_indiv = (1 - np.sum((count/total)**2, axis = 0))
    weights = np.sum(count, axis=0)/len(data_input.index)

    gini = np.sum(gini_indiv*weights)

    # print("Count:\n", data_input.value_counts(["temp_var", target]).sort_index())
    # print("Count:\n", count)
    # print(type(count), count.shape)
    # print("Total:\n", total)
    # print("Count/Total:\n", count/total)
    # print("Weight:\n", weights)
    # print("Gini individually:",gini_indiv)
    # print("Gini:",gini)
    # print(data.head())

    data.drop(['temp_var'], axis=1, inplace=True)

    return gini

#gini_calculation(data, 'test2', -0.454876, 'target') # nearly perfect split, seed: 20220919
gini_calculation(data, 'test2', -0.454876, 'target')

data_list = pd.DataFrame(columns = ['variable','unique_values','gini'])

for column in data:
    if column == 'target':
        continue
    print(column)
    temp = pd.DataFrame()
    temp['unique_values'] = pd.DataFrame(set(data[column]))
    temp['variable'] = column
    data_list = data_list.append(temp, ignore_index=True)

for index, row in data_list.iterrows():
    data_list.iloc[index, 2] = gini_calculation(data, row['variable'], row['unique_values'], 'target')

data_list = data_list.sort_values(by=['gini']).reset_index(drop=True)

print(temp.head())
print(data_list.head())
print(data_list.tail())

#SPLIT_VAR = str(data_list[0,0])
#SPLIT_NUM = data_list[0,1]

print(data_list.iloc[0,0])
print(data_list.iloc[0,1])

class1 = data[data[data_list.iloc[0,0]]<data_list.iloc[0,1]].target.mode()
class2 = data[data[data_list.iloc[0,0]]>=data_list.iloc[0,1]].target.mode()

print(data.head())
print(class1.head())
print(class2.head())

# validation
data_test = data.copy()
data_test['test'] = np.where(data_test[data_list.iloc[0,0]]<data_list.iloc[0,1], class1, class2)
print(data_test.head())

data_test['Correctly_Predicted'] = np.where(data_test['target'] == data_test['test'], True, False)
accuracy_ratio = data_test["Correctly_Predicted"].mean()

print(accuracy_ratio)