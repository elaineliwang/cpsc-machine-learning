#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:05:08 2018

@author: Elaine
"""

#basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# sklearn imports
#from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#our code
#import linear_model
import utils

def main():
    X = pd.read_csv('../data/BlackFriday.csv')# names =("User_ID", "Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status,", "Product_Category_1","Product_Category_2","Product_Category_3", "Purchase" ))
    N, d = X.shape
    print(N,d)
    # fill missing values with 0
    # (?) need to calculate percentage of missing value?
    X = X.fillna(0)
    # change gender to 0 and 1
    X['Gender'] = X['Gender'].apply(change_gender)
    # change age to 0 to 6
    X['Age'] = X['Age'].apply(change_age)
    # change city categories to 0 to 2
    X['City_Category'] = X['City_Category'].apply(change_city)
    # change the year to integer
    X['Stay_In_Current_City_Years'] = X['Stay_In_Current_City_Years'].apply(change_year)
    
    #predict gender
    y = np.zeros((N,1))    
    y = X.values[:,2]
    y = y.astype('int')
    X1 = X
    ID =['User_ID', 'Product_ID', 'Gender']
    X1 =  X1.drop(ID, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20,random_state=42)
    model = LogisticRegression(C = 1,  fit_intercept=False, solver='lbfgs',
                                   multi_class='multinomial')
    model.fit(X_train,y_train)
    print("LogisticRegression(softmax) Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("LogisticRegression(softmax) Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    
    model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)

    print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))


#predict the product category1  based on other information.
    y2 = np.zeros((N,1))    
    y2 = X.values[:,8]
    y2 = y2.astype('int')
    X2 = X
    ID =['User_ID' , 'Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    X2 =  X2.drop(ID, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2,random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5, metric = 'cosine')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    tr_error = np.mean(y_pred != y_train)

    y_pred = model.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("Training error of KNN to predict age: %.3f" % tr_error)
    print("Testing error of KNN to predict age: %.3f" % te_error)
# Training error of KNN to predict age: 0.363
#Testing error of KNN to predict age: 0.496
    
# Use decision tree to predict
    e_depth = 20
    s_depth = 1

    train_errors = np.zeros(e_depth - s_depth)
    test_errors = np.zeros(e_depth - s_depth)

    for i, d in enumerate(range(s_depth, e_depth)):
        print("\nDepth: %d" % d)

        model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        tr_error = np.mean(y_pred != y_train)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

        train_errors[i] = tr_error
        test_errors[i] = te_error

    x_vals = np.arange(s_depth, e_depth)
    plt.title("The effect of tree depth on testing/training error")
    plt.plot(x_vals, train_errors, label="training error")
    plt.plot(x_vals, test_errors, label="testing error")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.legend()

    fname = os.path.join("..", "figs", "trainTest_category1.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)
    
    
    model = RandomForestClassifier(criterion="entropy",n_estimators=5, max_features = 5)
    model.fit(X_train,y_train)
    print("RandomForest Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("RandomForest Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    #RandomForest Training error 0.027
    #RandomForest Validation error 0.157
    tree = DecisionTreeClassifier(max_depth=13, criterion='entropy', random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_train)
    tr_error = np.mean(y_pred != y_train)
    
    y_pred = tree.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    print("Decision Tree Training error : %.3f" % tr_error)
    print("Decision Tree Validation error: %.3f" % te_error)
    #Depth: 11
    #Training error: 0.127
    #Testing error: 0.131
    
    
    #use softmaxClassifier to predict occputation
    model = LogisticRegression(C = 10000,  fit_intercept=False, solver='lbfgs',
                                   multi_class='multinomial')
    model.fit(X_train,y_train)
    print("LogisticRegression(softmax) Training error %.3f" % utils.classification_error(model.predict(X_train), y_train))
    print("LogisticRegression(softmax) Validation error %.3f" % utils.classification_error(model.predict(X_test), y_test))
    #LogisticRegression(softmax) Training error 0.651
    #LogisticRegression(softmax) Validation error 0.652
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    from sklearn.metrics import mean_squared_error
    poly = PolynomialFeatures(degree=4)
    X_train_sub = X_train[:1000]
    y_train_sub = y_train[:1000]
    X_train_ = poly.fit_transform(X_train_sub)
    model = LinearRegression()
    model.fit(X_train_, y_train_sub)
    model.score(X_train_, y_train_sub, sample_weight=None)
    y_pred = model.predict(X_train_)
    tr_error = mean_squared_error(y_pred, y_train_sub)
    
    y_pred = model.predict(X_test) 
    te_error = np.mean(y_pred != y_test)
    print("Training error : %.3f" % tr_error)
    print("Validation error: %.3f" % te_error)
    
    
    #kernel = DotProduct() + WhiteKernel()
    y2 = np.zeros((N,1))    
    y2 = X.values[:,8]
    y2 = y2.astype('int')
    X2 = X
    ID =['User_ID' , 'Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
    X2 =  X2.drop(ID, axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.02,random_state=42)
    gpr = GaussianProcessRegressor(kernel=None,random_state=0).fit(X_train, y_train)
    gpr.score(X_train, y_train) 
    y_pred = gpr.predict(X_train)
    tr_error = mean_squared_error(y_pred,  y_train)
    y_pred = gpr.predict(X_test)
    te_error = mean_squared_error(y_pred, y_test)
    clf = KernelRidge(alpha=0.5)
    clf.fit(X_train_sub, y_train_sub) 
    clf.score(X_train_sub, y_train_sub, sample_weight=None)