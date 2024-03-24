from email.header import Header
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from pandas.api.types import is_numeric_dtype

# import data, skip first row "this is a header", delimiter is semicolon
data = pd.read_csv('/Users/meikeepauline/Desktop/Data/MLB_Ass3_dataa.csv',sep=';', header=None, skiprows=1)

# first row as column names
data.columns = data.iloc[0]

# remove first row (column names) and last row "this is a footer"
data = data[1:-1]

# print first rows of data
print(data.head())

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

# remove 'years from column age
data['age'] = data.age.str.replace('years','')

# column ash has value == 'missing' -> replace with "NaN"
data.ash = data.ash.replace('missing',np.nan)

# convert columns to type float, if error -> skipped
for i in range(18):
    data.iloc[:,i:i+1] = data.iloc[:,i:i+1].astype('float', errors = 'ignore')

# automatically convert datatype
data = data.convert_dtypes()
print("----------------------",data.dtypes)

# determine invalid data entries
print("------------ Unique values: ------------\n")
for col in data:
    print("------------",  str(col), "\n", data[col].unique(), "\n")

# set invalid data entries to null, correct inconsistent data entries for 'season'    
data.malic_acid = data.malic_acid.replace(-999,np.nan)
#data.proline = data.proline.replace(0,np.nan)
data.season = data.season.replace('spring','SPRING')
data.season = data.season.replace('aut','AUTUMN')

# find duplicate rows and delete
print(data[data.duplicated(keep=False)])
data.drop(labels=[179,180], axis=0, inplace=True)

# rename column od280/od315_of_diluted_wines to diluted_wines
data.rename(columns = {'od280/od315_of_diluted_wines':'diluted_wines'}, inplace = True)

# print statistics
print(data.total_phenols.describe())

#def histogram(var):
for col in data:
    sns.displot(data[col])
    # if is_numeric_dtype(var) == True:
    #     plt.hist(var[~np.isnan(var)])
    #     plt.title(str(var))
    # else:
    #     plt.hist(var)
    #     plt.title(str(var))

# histogram(data.alcohol)
# histogram(data.malic_acid)
# histogram(data.ash)
# histogram(data.alcalinity_of_ash)
# histogram(data.magnesium)
# histogram(data.total_phenols) #outliers
# histogram(data.flavanoids)
# histogram(data.nonflavanoid_phenols)
# histogram(data.proanthocyanins)
# histogram(data.color_intensity)
# histogram(data.hue)
# histogram(data.diluted_wines)
# histogram(data.proline)
# histogram(data.season)
# histogram(data.target)
# histogram(data.country)
# histogram(data.age)

plt.show()

data.proline = data.proline.replace(0,np.nan)
data['total_phenols'] = data['total_phenols'].where(data['total_phenols']<148, other=0)


#print(data.head())
#print(data.columns)
#print(data.dtypes)
#print(data.index)

# malic_acid -> -999

#alcohol, ash -> na

# proline -> 0

#season -> vereinheitlichen