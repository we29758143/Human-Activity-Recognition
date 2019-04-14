#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:06:42 2019

@author: lvguanxun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



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


#feeding differnet classifiers
#implementing validation data to see the performance
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = [
    KNeighborsClassifier(3), 
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
    ]

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
    
#implementing testing data and make predictions
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
    

test["result"] = train_predictions


#plot 3D image 
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


plotly.tools.set_credentials_file(username='w29758143', api_key='WifJOTMmpNLk2juo4Nqi')

data = []
clusters = []
colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)', 'rgb(160,32,240)','rgb(190,190,190)']

for i in range(len(test['result'].unique())):
    name = test['result'].unique()[i]

    color = colors[i]
    x = test[ test['result'] == name ]['roll_dumbbell']
    y = test[ test['result'] == name ]['pitch_dumbbell']
    z = test[ test['result'] == name ]['yaw_dumbbell']
    
    trace = dict(
        name = name,
        x = x, y = y, z = z,
        type = "scatter3d",    
        mode = 'markers',
        marker = dict( size=3, color=color, line=dict(width=0) ) )
    data.append( trace )

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='roll_dumbbell vs pitch_dumbbell vs yaw_dumbbell',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'        
    ),
)

fig = dict(data=data, layout=layout)
url = py.plot(fig, filename='dumbbell', validate=False)
