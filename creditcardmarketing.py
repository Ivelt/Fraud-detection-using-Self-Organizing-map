# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:53:13 2020

@author: ivelt
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('creditcardmarketing-bbm.csv')

clas=dataset.pop('OfferAccepted')
dataset.insert(16,'OfferAccepted',clas)



'''
print(dataset.isnull().sum().sum())
print(dataset.isnull().sum())

col = ["AverageBalance","Q1Balance","Q2Balance","Q3Balance","Q4Balance"]

for var in col:
    mean = dataset[var].mean()
    dataset[var].fillna(mean, inplace=True)
    '''
    
    
'''

dataset.columns=dataset.columns.str.replace(' ','_')

'''


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values


for i  in range(len(y)):
    if y[i]=='No':
        y[i]=0
    else:
        y[i]=1

    


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

numerical_features=['CustomerNumber','BankAccountsOpen','CreditCardsHeld','HomesOwned','HouseholdSize', 'AverageBalance','Q1Balance','Q2Balance','Q3Balance','Q4Balance']
categorical_features=['Reward','MailerType','IncomeLevel','OverdraftProtection','CreditRating','OwnYourHome']

numerical_pipeline=make_pipeline(SimpleImputer(strategy ='mean'),StandardScaler())
categorical_pipeline=make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder())
preprocessing=make_column_transformer((numerical_pipeline,numerical_features),(categorical_pipeline,categorical_features))

X=preprocessing.fit_transform(X)

X=np.delete(X,[10,13,15,18,20,23],1)


# Let's train the model now

from minisom import MiniSom
som = MiniSom(x=15,y=15,input_len=19)
som.random_weights_init(X)
som.train_random(X, num_iteration=500)

# Now let's visualize the representation

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

marker=['o','s']
color=['r','g']


for i, x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5,w[1]+0.5, marker[y[i]],
         markeredgecolor=color[y[i]],
         markerfacecolor='None',
         markersize=10, markeredgewidth=2)
show()
    
    
# Since there, We can map all the fraudulent applications 

Map = som.win_map(X)
fraudulentApplications = np.concatenate((Map[(2,4)], Map[(3,4)],Map[(3,5)],Map[(3,8)],Map[(2,8)],Map[(4,8)],Map[(7,4)],Map[(7,3)]),axis=0)

# there are 553 fraudulent applications ( we only consider the extreme cases)

len(fraudulentApplications)




''' 
    Here we are not going to just detect frauds  in the customer applications but instead 
    we are going to calculate the probability that a particular customer application might be fraudulent base on the whole records.
    And to do so, we are going to use artificial neuron networks
    
'''



customers=dataset.iloc[:,1:].values
is_fraud=np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in fraudulentApplications:
        is_fraud[i]=1
   
        

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,is_fraud,test_size=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu',input_dim=19))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam', metrics=['accuracy'],loss='binary_crossentropy')
classifier.fit(X_train, y_train, batch_size=100, epochs=5)

# prediction

prediction = classifier.predict(X_test)
pred=(prediction>0.5)

from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,pred)
c

max(prediction)






'''


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

Preprocess = make_column_transformer((StandardScaler(),['BankAccountsOpen','CreditCardsHeld','HomesOwned','HouseholdSize', 'AverageBalance','Q1Balance','Q2Balance','Q3Balance','Q4Balance'])
                                     (OneHotEncoder(),['Reward','MailerType','IncomeLevel','OverdraftProtection','CreditRating','OwnYourHome']))
                                    

                                 
X=Preprocess.fit_transform(X)
'''
'''
dataset.columns
df.columns = [x.strip() for x in df.columns] 
dataset.columns=[x.strip() for x in dataset.columns]
dataset.columns=dataset.columns.str.strip()

header(dataset)
names(dataset)
name(dataset)
dataset.columns
str(dataset.Average_Balance)
(dataset.Average_Balance).mean()
dataset.describe()
'''