import numpy as np
import sklearn, sklearn.tree
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

#
# creating a virtual environment in Python.
# python -m venv <virtual_environment_name>
# this will create the directory <virtual_environment_name>
# then activate the virtual environment:
# different for windows vs linux.  
# Using 'FredsVenv' as an example name:
# windows: ".\FredsVenv\Scripts\activate.bat"
# linux: "source ./FredsVenv/bin/activate.
# examine the code for activate and see what it is doing!
# when you are done playing, simply 'deactivate' the virtual environment.
#

## name = input("Hello, what's your name?  ")
## print("Well hello there, {}, lets get started.".format(name))
#
print("The sklearn version we are working with is {}".format(sklearn.__version__))

dataset = pd.read_csv('Chapter07/Social_Network_Ads.csv')
print( dataset.head(10) )

dataset = dataset.drop(columns=['User ID'])
print("After dropping User ID column, dataset: ")
print(dataset.head(10))

print( "Using pandas iloc to select all the rows from the first column of data: iloc[:, [0]]" )
print( dataset.iloc[:, [0]] )

#
# One-Hot Encode the Gender so it is a numeric value we can deal with 
#
encoder = sklearn.preprocessing.OneHotEncoder()
#
# iloc[:, [0]] selects all rows for the first column - gender for this dataset. 
# 
encoder.fit( dataset.iloc[:, [0]] )
onehotlabels = encoder.transform( dataset.iloc[:, [0]] ).toarray()
#
# check if you want, all ones and zeroes now, super cool.
#
# print( "One Hot Labels after toarray(): " )
# print(onehotlabels)

#
# now make a new dataframe from the labels array.
#
genders = pd.DataFrame( {'Female': onehotlabels[:, 0], 'Male': onehotlabels[:, 1]} )
print("The genders dataframe one-hot encoded in a new DataFrame: ")
print( genders.head(10) )

#
# now create a new dataframe from the original data without the genders column 
#
result = pd.concat( [genders, dataset.iloc[:, 1:]], axis=1, sort=False )
print("Result after concat() one-hot encoded gender and dropping the original gender column:")
print(result.head(10))

#
# now set up the actual training data and labels 
# y will be ground truth labels, just the purchased column
# X will be training data, the data without the purchased column
#
y = result['Purchased']
X = result.drop( columns=['Purchased'] )

#
# split the data into 75% training and 25% testing 
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("Now X_train info:")
print(X_train.info())

print("Now X_test info:")
print(X_test.info())

#
# Feature Normalization:
# scale the data into values between 0 and 1 before training
#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#
# the first fit_transform() calculates the means and standard deviations of the dataset,
# and then applies the scaling to the data using the calculated values
#
X_train = scaler.fit_transform(X_train)
#
# the second scaler just uses transform(), which uses the values calculated from the
# previous call to fit_transform() - does not recalculate mean/std deviation, to prevent
# data leakage that could happen with different values in the testing dataset.
#
X_test = scaler.transform(X_test)

#
# the scaler also converts the pandas dataset into a numpy.ndarray!
#
print("After scaling, our smaller testing dataset now looks like: ")
print(X_test)

#
# train a Decision Tree classifier
#
decision_tree_classifier = sklearn.tree.DecisionTreeClassifier(
        criterion='entropy', random_state=100, max_depth=2)

decision_tree_classifier.fit(X_train, y_train)

#
# now use the trained classifier to predict labels for the testing data
#
y_pred = decision_tree_classifier.predict(X_test)

#
# now get some metrics to tell us how well the decision tree classifier did 
# with the parameters we used to initialize
#
import sklearn.metrics as metrics
decision_tree_metrics = metrics.confusion_matrix(y_test, y_pred)
print("Decision Tree Classifier Confusion Matrix: ")
print( decision_tree_metrics )

#
# Finally calculate the accuracy, recall, and precision 
# for the decision tree classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "Decision Tree Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))

#
# Now Introducing XGBoost Classifier
#
from xgboost import XGBClassifier
#
# XGBoost hyperparameter dictionary
# change max_depth to 2 and it matches the decision tree!
#
xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'learning_rate': 0.1,
        'n_estimators': 50}
xgb_classifier = XGBClassifier(**xgb_params)
xgb_classifier.fit(X_train, y_train)
print("XGB Classifier: ")
print(xgb_classifier.get_params())
y_pred = xgb_classifier.predict(X_test)
xgb_metrics = metrics.confusion_matrix(y_test, y_pred)
print("XGBoost Confusion Matrix: ")
print( xgb_metrics )


#
# Finally calculate the accuracy, recall, and precision 
# for the XGBoost classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "XGBoost Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))

#
# Random Forest Classifier
#
from sklearn.ensemble import RandomForestClassifier
#
# random forest hyperparameters
#
rand_forest_params = {
        'n_estimators': 10,
        'max_depth': 3,
        'criterion': 'entropy',
        'random_state': 0}

print("Random Forest Classifier: ")
random_forest_classifier = RandomForestClassifier(**rand_forest_params)
print(random_forest_classifier.get_params())
random_forest_classifier.fit(X_train, y_train)
y_pred = random_forest_classifier.predict(X_test)
random_forest_metrics = metrics.confusion_matrix(y_test, y_pred)
print("Random Forest Confusion Matrix:")
print(random_forest_metrics)
#
# Finally calculate the accuracy, recall, and precision 
# for the Random Forest Classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "Random Forest Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))


#
# Logistic Regression Classifier
#
from sklearn.linear_model import LogisticRegression
logistic_regression_classifier = LogisticRegression(random_state=0)
print("Logistic Regression Classifier: ")
print(logistic_regression_classifier.get_params())
logistic_regression_classifier.fit(X_train, y_train)
y_pred = logistic_regression_classifier.predict(X_test)
logistic_regression_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Logistic Regression Confusion Matrix:")
print(logistic_regression_confusion_matrix)
#
# Finally calculate the accuracy, recall, and precision 
# for the Logistic Regression Classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "Logistic Regression Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))


#
# Support Vector Machine Classifier
#
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', random_state=0)
print("Support Vector Machine Classifier: ")
print(svm_classifier.get_params())
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
svm_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Support Vector Machine Confusion Matrix:")
print(svm_confusion_matrix)
#
# Finally calculate the accuracy, recall, and precision 
# for the Logistic Regression Classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "Support Vector Machine Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))















        



