# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:32:24 2016

@author: chirag
"""


from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score,KFold
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer



os.chdir('C:\\Users\\chirag\\Desktop\\Data Science\\final')

data_file=datasets.load_boston()
predictors=data_file.data
target=data_file.target


'''
- CRIM     per capita crime rate by town\n    
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n  
- INDUS    proportion of non-retail business acres per town\n        
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        
- NOX      nitric oxides concentration (parts per 10 million)\n        
- RM       average number of rooms per dwelling\n        
- AGE      proportion of owner-occupied units built prior to 1940\n        
- DIS      weighted distances to five Boston employment centres\n        
- RAD      index of accessibility to radial highways\n        
- TAX      full-value property-tax rate per $10,000\n        
- PTRATIO  pupil-teacher ratio by town\n        
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        
- LSTAT    % lower status of the population\n        
- MEDV     Median value of owner-occupied homes in $1000's\n\
'''


# Ans 1


depth=[1,2,5,10]
test=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def depth_train_errorplot():
    for d in depth:
        MSE_train=[]
        MSE_test=[]
        for t in test: 
            train_X, test_X, train_Y, test_Y =  train_test_split(predictors,target, test_size =t, random_state = 99)    
            regressor = DecisionTreeRegressor(max_depth=d,random_state=99)
            regressor.fit(train_X,train_Y)
            predicted_train=regressor.predict(train_X)
            predicted=regressor.predict(test_X)
            MSE_train_temp=metrics.mean_squared_error(train_Y,predicted_train)
            MSE_test_temp=metrics.mean_squared_error(test_Y,predicted)
            MSE_train.append(MSE_train_temp)
            MSE_test.append(MSE_test_temp)
        plt.xlabel('Testing %')
        plt.ylabel('MSE')
        plt.plot(test, MSE_train)
        plt.plot(test, MSE_test )
        plt.legend(['MSE train','MSE test'], loc='upper left')
        plt.title('Train Test MSE Errors')
        plt.show()


depth_train_errorplot()


# Ans 2


depth2=range(1,13)
def depth_wise_error():
    MSE_train=[]
    MSE_test=[]
    for d in depth2: 
        train_X, test_X, train_Y, test_Y =  train_test_split(predictors,target, test_size =.4, random_state = 99)    
        regressor = DecisionTreeRegressor(max_depth=d,random_state=99)
        regressor.fit(train_X,train_Y)
        predicted_train=regressor.predict(train_X)
        predicted=regressor.predict(test_X)
        MSE_train_temp=metrics.mean_squared_error(train_Y,predicted_train)
        MSE_test_temp=metrics.mean_squared_error(test_Y,predicted)
        MSE_train.append(MSE_train_temp)
        MSE_test.append(MSE_test_temp)
    plt.xlabel('Depth %')
    plt.ylabel('MSE error')
    plt.plot(depth2, MSE_train)
    plt.plot(depth2, MSE_test )
    plt.legend(['MSE train','MSE test'], loc='upper right')
    plt.title('Train Test MSE Errors')
    plt.show()


depth_wise_error()

# Ans 3


def predict_using_grid_searchCV():
    train_sample=[[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
    
    train_X, test_X, train_Y, test_Y =  train_test_split(predictors,target, test_size =.4, random_state = 99)    
    
    sc = make_scorer(mean_squared_error, greater_is_better=False)
    parameters = {'max_depth':(8,9,10,11,12,13,14,15,16,17,18,19),'max_features':(5,6,7,8,9,10,11),'splitter': ('best','random')}
    
    regressor = DecisionTreeRegressor(random_state=99)
    reg = grid_search.GridSearchCV(regressor, parameters, cv=10,scoring=sc)
    reg.fit(predictors,target)
    predicted_train=reg.predict(train_X)
    predicted=reg.predict(test_X)
    MSE_train=metrics.mean_squared_error(train_Y,predicted_train)
    MSE_test=metrics.mean_squared_error(test_Y,predicted)
    print("MSE_train : ",MSE_train,"MSE_test : ",MSE_test)
    predicted_sample=reg.predict(train_sample)
    
    best_parameter=reg.best_estimator_ 
    print("best parameters:",best_parameter)
    print(predicted_sample)
    

predict_using_grid_searchCV()

















