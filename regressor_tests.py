import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

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
#
# to install dependencies with the correct versions,
# activate the virtual environment and 
# pip install -r classification_tests_requirements.txt
# 
# when you are done playing, simply 'deactivate' the virtual environment.
#

print("The sklearn version we are working with is {}".format(sklearn.__version__))
_ = input('Press Enter to continue.')

dataset = pd.read_csv('auto.csv')
print( dataset.head(10) )

dataset = dataset.drop(columns=['NAME'])
print("After dropping NAME column, dataset: ")
print(dataset.head(10))

#
# convert all the input variables and impute any null values in the dataset
#
dataset = dataset.apply(pd.to_numeric, errors = 'coerce')
dataset.fillna(0, inplace=True)


#
# set up the actual training data and labels 
# y will be ground truth labels, just the MPG column
# X will be training data, the data without the MPG column
#
y = dataset['MPG']
X = dataset.drop( columns=['MPG'] )

#
# split the data into 75% training and 25% testing 
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("Now X_train info:")
print(X_train.info())

print("Now X_test info:")
print(X_test.info())

print("Now y_train info:")
print(y_train.info())

print("Now y_test info:")
print(y_test.info())


#
# Linear Regression 
#
linear_regression = LinearRegression()
print("\nLinear Regression Parameters: ")
print(linear_regression.get_params())
linear_regression.fit(X_train, y_train)

#
# now use the trained classifier to predict labels for the testing data
#
y_pred = linear_regression.predict(X_test)

#
# and now get the Root Mean Squared Error for Linear Regression
# 
linear_regression_rmse = sqrt( mean_squared_error(y_test, y_pred) )
print( "Linear Regression Root Mean Squared Error: ", end = '')
print( linear_regression_rmse )


#
# Decision Tree Regressor
#
decision_tree_regressor = DecisionTreeRegressor(max_depth=3)

print("\nDecision Tree Regressor Parameters: ")
print(decision_tree_regressor.get_params())

decision_tree_regressor.fit(X_train, y_train)
y_pred = decision_tree_regressor.predict(X_test)

dt_regressor_rmse = sqrt( mean_squared_error(y_test, y_pred) )
print( "Decision Tree Regressor Root Mean Squared Error: ", end = '')
print( dt_regressor_rmse )


#
# Gradient Boost Regression
#
# The 'loss' parameter of GradientBoostingRegressor must be a str among {'absolute_error', 'quantile', 'huber', 'squared_error'}.
#
gb_params = {
        'n_estimators': 500,
        'max_depth': 4,
        'min_samples_split': 2,
        'learning_rate': 0.01,
        'loss': 'squared_error' }    # rmse of 4.034623529237156
        # 'loss': 'huber' }    # rmse of 4.178821955008476
        # 'loss': 'quantile' }    # rmse of  6.108262123116929
        #'loss': 'absolute_error' } # rmse of 4.319324880182872

gradient_boost_regressor = ensemble.GradientBoostingRegressor(**gb_params)

print("\nGradient Boosting Regressor Parameters: ")
print(gradient_boost_regressor.get_params())

gradient_boost_regressor.fit(X_train, y_train)
y_pred = gradient_boost_regressor.predict(X_test)

gradient_boost_rmse = sqrt( mean_squared_error(y_test, y_pred) )
print("Gradient Boost Regressor Root Mean Squared Error: ", end = '')
print(gradient_boost_rmse)



