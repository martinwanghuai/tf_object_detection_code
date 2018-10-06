'''
Created on 4 Oct 2018

@author: martinwang
'''
import numpy as np
import pandas as pd
import pandas.tools as tool
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')

print(dataset.shape)
print(dataset.size)
print(dataset.info())
print(dataset.describe())
print(dataset.head())
print(dataset.tail())
print(dataset.sample(5))
dataset = dataset.drop('Id', axis = 1)

# print(dataset['Species'].unique())
# print(dataset['Species'].value_counts())
# print(dataset.isnull().sum())
# dataset = dataset.drop('Id', axis=1)
# print(dataset.groupby('Species').count())
# print(dataset[dataset['SepalLengthCm']>7.2])
# print(dataset.where(dataset['SepalLengthCm']>7.2)) #not so good

#Scatter plot
# sns.FacetGrid(dataset, hue="Species", size=5) \
#     .map(plt.scatter, "SepalLengthCm", "SepalWidthCm") \
#     .add_legend()
# plt.show()  

#Box plot to show distribution
# dataset.plot(kind='box', subplots = True, layout = (2,2), sharex = False, sharey= False)
# plt.show()

#Box plot to show relevance 1
# sns.boxplot(x='Species', y = 'PetalLengthCm', data=dataset)
# plt.show()

#Box plot to show relevance 2
# ax = sns.boxplot(x='Species', y='PetalLengthCm', data= dataset)
# ax = sns.stripplot(x='Species', y='PetalLengthCm', data=dataset, jitter=True, edgecolor='gray')
# plt.show()

#Box plot to show relevance 3
# ax = sns.boxplot(x='Species', y='PetalLengthCm', data=dataset)
# ax = sns.stripplot(x='Species', y='PetalLengthCm', data=dataset, jitter=True, edgecolor='gray')
# boxtwo = ax.artists[1]
# boxtwo.set_facecolor('red')
# boxtwo.set_edgecolor('black')
# boxthree = ax.artists[2]
# boxthree.set_facecolor('yellow')
# boxthree.set_edgecolor('black')
# plt.show()

#histograms
# dataset.hist(figsize=(10, 5))
# plt.show()

#multivariate plots : important since it shows correlation between features
# import pandas.plotting as plotting
# plotting.scatter_matrix(dataset, figsize=(10,5))
# plt.show()

#violinplots
# sns.violinplot(data=dataset, x='Species', y='PetalLengthCm')
# plt.show()

#pairplot: important since it shows correlation between features and labels
# sns.pairplot(data=dataset,hue='Species', diag_kind='kde')
# plt.show()

#kdeplot: kernal dentisity estimation
# sns.FacetGrid(dataset, hue='Species', size=5) \
#     .map(sns.kdeplot, 'PetalLengthCm') \
#     .add_legend()
# plt.show()    

#jointplot
# sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', 
#               data=dataset, size=10, ratio=10, kind='hex', color='green')
# plt.show()
# sns.jointplot(x='SepalLengthCm', y='SepalWidthCm',
#               data=dataset, height=5, ratio=5, kind='kde', color='#800000', space=0)
# plt.show()


#andrews curves
# from pandas.tools.plotting import andrews_curves
# andrews_curves(dataset, 'Species', colormap='rainbow')
# plt.show()

#heatmap: important since it shows Pearson correlation coefficiency between features
# plt.figure(figsize=(7,4))
# sns.heatmap(dataset.corr(), annot=True, cmap='cubehelix_r')
# plt.show()

#radiviz: 
from pandas.tools.plotting import radviz
radviz(dataset, 'Species')
plt.show()


