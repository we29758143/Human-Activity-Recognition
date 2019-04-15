# Human-Activity-Recognition

# 1.	Overview
*********************
Human Activity Recognition - HAR - has been recognized as a key research area and is gaining attention by the computing   research community, especially for the development of context-aware systems. There are many potential applications for   		HAR, like: elderly monitoring, life log systems for monitoring energy expenditure and for supporting weight-loss programs,and digital assistants for weight lifting exercises.
		
# 2.	Background

The goal of this assignment is for candidate predict the manner in which they performed the exercise and machine learning classification of accelerometers data on the belt, forearm, arm, and dumbbell of 6 participants. In training data “classe” is the outcome variable in the training set using predictor variables to predict 20 different test cases.

# 3.	Prediction Modeling

First, Preprocess the data. I drop the data which is nan. The original training data have 160 features, after dropping feature, it have 60 features. Also, using one hot encoder to process the feature "new_window" which is "yes" or "no". And drop the feature which is time based.

Second, I used several kind of classifiers to see which gave the best performance. I used KNN, Ada Boost, Gradient Boosting, Gaussian NB, Decesion tree and Random Forest as our classifiers.


Thrid, split the training data into training data and validation, which proportion is 9:1(19622:1963). Then implement into the classifiers.

## Result:

### KNeighborsClassifier

Accuracy: 36.9333%

Log Loss: 12.266747496440916

### AdaBoostClassifier

Accuracy: 71.0138%

Log Loss: 1.382516469578437

### GradientBoostingClassifier

Accuracy: 99.3377%

Log Loss: 0.08941841346737066

### GaussianNB

Accuracy: 53.5405%

Log Loss: 1.6245191855954215

### DecisionTreeClassifier

Accuracy: 98.7774%

Log Loss: 0.42227744955571334

### RandomForestClassifier

Accuracy: 99.6943%

Log Loss: 0.07236775655903403

Decision: Random Forest classifier gave the best performance.



# 4.	Data Exploration
		
There are several features which will highly influence the predictions. In this part, I used random forest classifier as   our classifier since it gave us the highest accuracy on the validation data. 

## Accel_Belt
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/accel_belt.png)

URL of 3D plot: https://plot.ly/~w29758143/2/accel-belt/#/

## Gyros_Belt
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/gyros_belt.png)

URL of 3D plot: https://plot.ly/~w29758143/6/gyros-belt/#/

## Magnet_Belt
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/magnet_belt.png)

URL of 3D plot: https://plot.ly/~w29758143/8/magnet-belt/#/

## total_accel_belt vs total_accel_arm vs total_accel_forearm
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/total_accel.png)

URL of 3D plot: https://plot.ly/~w29758143/8/total-accel-vs-total-accel-arm-vs-total-accel-forearm/#/


## roll_belt vs pitch_belt vs yaw_belt
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/belt%20(1).png)

URL of 3D plot: https://plot.ly/~w29758143/10/roll-belt-vs-pitch-belt-vs-yaw-belt/#/

## roll_arm vs pitch_arm vs yaw_arm
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/arm.png)

URL of 3D plot: https://plot.ly/~w29758143/12/roll-arm-vs-pitch-arm-vs-yaw-arm/#/

## roll_forearm vs pitch_forearm vs yaw_forearm
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/forearm.png)

URL of 3D plot: https://plot.ly/~w29758143/14/roll-forearm-vs-pitch-forearm-vs-yaw-forearm/#/


## roll_dumbbell vs pitch_dumbbell vs yaw_dumbbell
![image](https://github.com/we29758143/Human-Activity-Recognition/blob/master/dumbbell.png)

URL of 3D plot: https://plot.ly/~w29758143/16/roll-dumbbell-vs-pitch-dumbbell-vs-yaw-dumbbell/#/

Summary: The interesting part of the result is I found out that the x axis, y axis and z axis did not play an important role of predicting the class. No matter it is accelemeter or gyroscope. What gave us more intuition of predicting the class is based on roll, pitch and yaw. Although class "A" and "B" dominate the prediction. We can see that the last four figures make a clear boundry which cluster the same class.

# 5.	Model Application
The reason I used several classifier not neural network is because Randon Forest Classifier already gave us really high accuracy on validation data. 
We can use this model to predict several type of use case. Not only in this regard, we can put more device on body. Then imporve athlete performance such as basketball shooting, scoccer, swimming, etc. For example, if we can retrieve a data from an excellent basketball shooter then we can try to understand how the movement of the body will affect shooting rate. 	
	ffffffffffff
