# Assignment: 
# import iris data, which contains 4 features (sepal length & width, petal length & width) and target (setosa, versicolor, virginica) 
# train kNN-model and make predictions


# import packages
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# load iris data and create dataframe
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['target']=iris.target

# print data and target names
print(df)
print(iris.target_names)

# analyse data
stats_df = df.loc[0:,:'petal width (cm)'].describe().transpose()
print(stats_df)

# prepare data for KNeighbors model
# 4 variables (sepal length and width, petal length and width) as X (Features)
# 1 variable (target) as Y (Target)
X_train = df.drop(['target'], axis=1)
Y_train = df['target']

# Selected Algorithm: k nearest neighbours with k = 3
kNClass = KNeighborsClassifier(n_neighbors=3)

# Train model with full dataset
# .values necessary or UserWarning is triggered
kNClass.fit(X_train.values, Y_train.values)

# Make predictions:
# 3x with values from sample - first row in data of each targetclass
# 1x with random values
print(df.groupby('target').first())

def prediction(values):
    Y_pred = kNClass.predict([values])
    print("Prediction of", [values], " is: ", Y_pred, iris.target_names[Y_pred])

prediction([5.1, 3.5, 1.4, 0.2])
prediction([7.0, 3.2, 4.7, 1.4])
prediction([6.3, 3.3, 6.0, 2.5])
prediction([5.0, 3.1, 2.3, 2.0])