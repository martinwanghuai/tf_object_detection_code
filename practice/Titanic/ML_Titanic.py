'''
Created on 4 Oct 2018

@author: martinwang
'''
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print(test.head())

print('Total number of passengers in the training data...', len(train))
print('Number of passengers in the training data who survived...', len(train[train['Survived']==1]))

print(train['Survived'][train['Sex']=='male'])

print('% of men who survived', 100*np.mean(train['Survived'][train['Sex']=='male']))
print('% of women who survived', 100*np.mean(train['Survived'][train['Sex']=='female']))


print('% of passengers who survived in first class', 100*np.mean(train['Survived'][train['Pclass'] ==1]))
print('% of passengers who survived in third class', 100*np.mean(train['Survived'][train['Pclass'] ==3]))

print('% of children who survived', 100*np.mean(train['Survived'][train['Age']<18]))
print('% of adults who survived', 100*np.mean(train['Survived'][train['Age']>=18]))

train['Sex'] = train['Sex'].apply(lambda x:1 if x=='male' else 0) #One-hot

train['Age'] = train['Age'].fillna(np.mean(train['Age'])) # missing value
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare'])) #missing value

train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

y = train['Survived']
X = train.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))
print('Validation accuracy...', accuracy_score(y_test, classifier.predict(X_test)))

from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

def visualize(classifier, X_test):
    with open("tree1.dot",'w') as f:
        f = tree.export_graphviz(classifier,
                             out_file=f,
                             impurity=False,
                             feature_names = X_test.columns.values,
                             class_names = ['No','Yes'],
                             rounded = True,
                             filled = True)

        check_call(['dot','-Tpng','tree1.dot', '-o', 'tree1.png'])

    img = Image.open('tree1.png')
    draw = ImageDraw.Draw(img)
    img.save('sample-out.png')
    PImage('sample-out.png')
    
classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)
visualize(classifier, X_test)