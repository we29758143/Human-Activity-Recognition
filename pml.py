#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:06:42 2019

@author: lvguanxun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test = pd.read_csv("pml-testing.csv")
train = pd.read_csv("pml-training.csv")



train = train.dropna(axis = 1) #drop nan data
test = test.dropna(axis = 1)  #drop nan data

#preprossing training data
train_label = train["classe"]
one_hot_new_window = pd.get_dummies(train["new_window"]) #use one hot encoder to process new_window
train_data = train.drop(columns = ["Unnamed: 0", "user_name", "cvtd_timestamp","classe", "new_window"])

train_data["new_window_no"] = one_hot_new_window["no"]
train_data["new_window_yes"] = one_hot_new_window["yes"]

#preprossing testing data
one_hot_new_window_test = pd.get_dummies(test["new_window"]) #use one hot encoder to process new_window
test_data = test.drop(columns = ["Unnamed: 0", "user_name", "cvtd_timestamp","problem_id", "new_window"])

test_data["new_window_no"] = one_hot_new_window_test["no"]

zero_data = np.zeros(shape=(20,1))  # given new_window_yes all zero since it did not contain yes in this feature
test_data["new_window_yes"] = zero_data

#split training data into training data and validation data portion 1:9
from sklearn.model_selection import train_test_split
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(train_data, train_label, test_size=0.1, random_state=42)


#implementing validation data to see the performance
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3), 
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    DecisionTreeClassifier()
    ]


log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")
print("================Validation==============================")

for clf in classifiers:
    clf.fit(X_train_val, y_train_val)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test_val)
    acc = accuracy_score(y_test_val, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions_prob = clf.predict_proba(X_test_val)
    ll = log_loss(y_test_val, train_predictions_prob)
    print("Log Loss: {}".format(ll))
    

print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")

for clf_test in classifiers:
    clf_test.fit(train_data, train_label)
    name = clf_test.__class__.__name__
    
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf_test.predict(test_data)
    
    print("Prediction: ",train_predictions)
    
 

print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
print("================Testing data==============================")
    


