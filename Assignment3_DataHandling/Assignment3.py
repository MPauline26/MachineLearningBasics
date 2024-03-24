from email.header import Header
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

data = pd.read_csv('/Users/meikeepauline/Desktop/Data/MLB_Ass3_dataa.csv',sep=';', header=None, skiprows=1)


print(data.head())

data.columns = data.iloc[0] 
data = data[1:-1]

print(data.head())

#data = data.convert_dtypes()

print("Unique: ", data.alcohol.unique())
print("teeeeest", data[data.ash == 'missing'])
print("potato\n", data.loc[~data['alcohol'].astype(str).str.isdigit()])

def comma_to_dot(var):
    #data[var] = data[var].str.replace(',','.').astype('float')
    data[var] = data[var].astype('float')

#test = data[data['total_phenols'] == '2,45']

for i in range(17):
    data.iloc[51:52,i] = data.iloc[51:52,i].str.replace(',','.')

for x in range(1,17):
    if data.iloc[142:143,x].isna().item() == False:
        print(x)
        data.loc[142:143,'alcohol'] = data.loc[143:143,'alcohol'] + ',' + data.iloc[142:143,x]
        print('VERFICKT', data.loc[142:143,'alcohol'])
        data.iloc[142:143,x] = ''

data.iloc[50:51,0:17] = data.iloc[50:51,0].str.split(pat=",",expand=True)
data.iloc[142:143,0:17] = data.iloc[142:143,0].str.split(pat=",",expand=True)
data.iloc[166:167,4:6] = data.iloc[166:167,4].str.split(pat=" ",expand=True)

data.ash = data.ash.replace('missing',np.nan)

# alternative: data.loc[142:143,'alcohol'] = data.iloc[142:143,0] + ',' + data.iloc[142:143,1]

#data.loc[142:143,:] = 
#test = pd.DataFrame(data.iloc[142:143,1:].shift(periods=5,axis="columns"))
#data.iloc[142:143,1:17] = test
#print("FUUUCK", test, "\n", data.iloc[142:144,])
#data.iloc[50:51,0:17] = data.iloc[50:51,0].str.split(pat=",",expand=True)
#data.iloc[142:143,1:17].shift(periods=5, axis="columns")

comma_to_dot('alcohol')
comma_to_dot('malic_acid') #replace , with .
comma_to_dot('ash') #--remove 'missing'
comma_to_dot('alcalinity_of_ash') #replace , with .
comma_to_dot('magnesium') #--remove '111 1.7'
comma_to_dot('total_phenols') #replace , with .
comma_to_dot('flavanoids') #replace , with .
comma_to_dot('nonflavanoid_phenols') #replace , with .
comma_to_dot('proanthocyanins') #replace , with .
comma_to_dot('color_intensity') #replace , with .
comma_to_dot('hue') #replace , with .
comma_to_dot('od280/od315_of_diluted_wines') #replace , with .
comma_to_dot('proline') #no problems
comma_to_dot('color') #no problems
comma_to_dot('target') #no problems

data2 = data.convert_dtypes()
print("----------------------",data2.dtypes)

#for i in range(15):
#    data.iloc[:,i] = data.iloc[:,i].astype('float')

print("HEEEERE\n", data.iloc[50:53,],"\n", data.iloc[140:144,],"\n", data.iloc[133:140,])
print(data.head())
print(data.columns)
print(data.dtypes)
print(data.index)