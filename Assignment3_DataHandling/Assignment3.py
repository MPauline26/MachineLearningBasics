# to surpress FutureWarning message
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import KNNImputer

# to see all info of dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# import data, skip first row "this is a header", delimiter is semicolon
data = pd.read_csv('/Users/meikeepauline/Desktop/Data/MLB_Ass3_dataa.csv',sep=';', header=None, skiprows=1)

# first row as column names
data.columns = data.iloc[0]

# remove first row (column names) and last row "this is a footer"
data = data[1:-1]

# print first rows of data and some statistics
print(data.head())
print(data.info())
print(data.describe())

# row 52 has commas as decimal separator -> change to dot
for i in range(17):
    data.iloc[51:52,i] = data.iloc[51:52,i].str.replace(',','.')

# row 143 has commas and semi colon as delimiter -> concatenate all values to one column
for x in range(1,17):
    if data.iloc[142:143,x].isna().item() == False:
        data.loc[142:143,'alcohol'] = data.loc[143:143,'alcohol'] + ',' + data.iloc[142:143,x]
        data.iloc[142:143,x] = ''

# split data into multiple columns, where it was not correctly done 
data.iloc[50:51,0:17] = data.iloc[50:51,0].str.split(pat=",",expand=True)
data.iloc[142:143,0:17] = data.iloc[142:143,0].str.split(pat=",",expand=True)
data.iloc[166:167,4:6] = data.iloc[166:167,4].str.split(pat=" ",expand=True)

# split coulmn "country-age" to two columns and drop
data[['country','age']] = data['country-age'].str.split(pat="-",expand=True)
data.drop('country-age', axis=1, inplace=True)

# remove 'years' from column age
data['age'] = data.age.str.replace('years','')

# column ash has value == 'missing' -> replace with "NaN"
data.ash = data.ash.replace('missing',np.nan)

# convert columns to type float, if error -> skipped
for i in range(18):
    data.iloc[:,i:i+1] = data.iloc[:,i:i+1].astype('float', errors = 'ignore')
print("Datatypes: ",data.dtypes)

# determine invalid data entries
print("------------ Unique values: ------------\n")
for col in data:
    print("------------",  str(col), "\n", data[col].unique(), "\n")

# set invalid data entries to null, adjust inconsistent data entries for 'season'    
data.malic_acid = data.malic_acid.replace(-999,np.nan)
data.season = data.season.replace('spring','SPRING')
data.season = data.season.replace('aut','AUTUMN')

# find duplicate rows and delete
print("Duplicates: ",data[data.duplicated(keep=False)])
data.drop(labels=[179,180], axis=0, inplace=True)

# rename column od280/od315_of_diluted_wines to diluted_wines
data.rename(columns = {'od280/od315_of_diluted_wines':'diluted_wines'}, inplace = True)

# plot histogram
for col in data:
    sns.displot(data[col])

# boxplot for column 'magnesium' due to outliers
sns.boxplot(data=data[['magnesium']])
plt.show()

# set outliers to np.nan (99.Quantile)
data.proline = data.proline.replace(0,np.nan)
data['total_phenols'] = data['total_phenols'].where(data['total_phenols']<np.percentile(data.total_phenols, 99), other=np.nan)
data['magnesium'] = data['magnesium'].where(data['magnesium']<np.percentile(data.magnesium, 99), other=np.nan)
 
# detect columns with missing values
print("Missing values: ", data.isna().sum())

# import kNN Imputer
imputer = KNNImputer(n_neighbors=5)

# remove columns, which are not in original load_wine() dataset and target (should never be used for imputation)
data_impute = data.copy()
data_impute.drop(['color','age','country','season', 'target'], axis=1, inplace=True)

# impute data
data_impute = pd.DataFrame(imputer.fit_transform(data_impute),columns=data_impute.columns)
print("Missing values after Imputing: ", data_impute.isna().sum())

# add dropped columns back to data set
data = data[['color','age','country','season','target']]
data = data.reset_index(drop=True)
data_fin = pd.concat([data_impute, data], axis=1)
print(data_fin.head())

# determine class distribution
class_distr = data_fin.groupby("target").describe()
print(class_distr)

# group magnesium by color and calculate statistics within groups
magnesium = data_fin.groupby("color")["magnesium"].describe()
print(magnesium)

# plot for fixed dataset
sns.pairplot(data_fin, height = 2.0, hue = 'target')
plt.savefig('pairplot_fixed_data.png')

# plot:
# - a scatterplot for each variable pair -> correlation for each target class
# - distribution for every variable per target class

# result:
# flavanoids might be a good variable to separate the three classes
# age and color are bad discriminators
# flavanoids and total_phenols show a positive correlation
# proline and diluted_wines might be a good discriminator pair according to scatter plot
# magnesium and ash show nearly no correlation and separation ability together