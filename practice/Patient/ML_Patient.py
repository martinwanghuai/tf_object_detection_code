'''
Created on 8 Oct 2018

@author: martinwang
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas.plotting as plotting

dataset = pd.read_csv('indian_liver_patient.csv')

print(dataset.describe(include='all'))
print(dataset.columns)
print(dataset['Dataset'].value_counts())

#important: can show is null values
print(dataset.isnull().sum())

#important: drop rows with null values
dataset = dataset.dropna(how='any', axis=0) 

#important: convert "Gender" and "Dataset" labels
dataset['Gender'] = dataset['Gender'].map({'Male':0, 'Female':1})
dataset['Dataset'] = dataset['Dataset'].map({2:0, 1:1}) 

# print(dataset.sample(10))

#important: show the correlation among variables
lp_corr = dataset.drop(['Gender', 'Dataset'], axis=1)
print(lp_corr.corr())

#important: how to split rows
rows = dataset.shape(0)
train_count = int(0.8*rows)
test_count = rows - train_count

train_data = dataset[:train_count]
test_data=dataset[train_count:]

#

# dataset.hist(figsize=(10,5))
# plt.show()

# plotting.scatter_matrix(dataset,figsize=(20,10))
# plt.show()

# plt.figure(figsize=(20,10))
# sns.heatmap(dataset.corr(), annot=True, cmap='cubehelix_r')
# plt.show()

# dataset.plot(kind='box', subplots = True, layout = (3,4), sharex = False, sharey= False)
# plt.show()

# sns.pairplot(dataset, hue='Dataset', diag_kind='kde')
# plt.show()
