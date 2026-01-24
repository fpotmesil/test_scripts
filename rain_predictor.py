import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Chapter07/weather.csv')

print("Read weather.csv, the shape of our dataset: ", sep = '')
print(dataset.shape)

print("Columns of weather.csv:")
print(dataset.columns)

print("Ok, lets look at the first ten rows of the first 12 columns with iloc[:, 0:12].head(10)")
print( dataset.iloc[:, 0:12].head(10) )

print("Now the first ten rows of the last columns with iloc[:, 12:30].head(10)")
#
# iloc[:, 12:30] wont throw and exception, just dumps them all.
# probably cleaner to use 12:23 since we know the number of columns from the dataset.shape
# cleaner even still to use dataset.shape[1] to get the number of columns.
#
print( dataset.iloc[:, 12:dataset.shape[1]].head(10) )

print("Lets look at WindGustDir value_counts(): ")
print( dataset.WindGustDir.value_counts() )
print("Lets look at WindGustDir value_counts().index: ")
print( dataset.WindGustDir.value_counts().index )
print("Lets look at WindGustDir value_counts().values: ")
print( dataset.WindGustDir.value_counts().values )

print("Lets look at WindDir9am value_counts(): ")
print( dataset.WindDir9am.value_counts() )

print("Lets look at WindDir3pm value_counts(): ")
print( dataset.WindDir3pm.value_counts() )

#
# FJP TODO - get the values from value_counts and sort them properly,
# then get all on a graph together to see what it looks like.
#
from matplotlib import pyplot as plt
dataset.WindGustDir.value_counts().plot(kind = 'barh', color = 'red', label='Wind Gust Direction')
plt.show()

dataset.WindDir9am.value_counts().plot(kind = 'barh', color = 'blue', label='Wind Direction 9am')
plt.show()

dataset.WindDir3pm.value_counts().plot(kind = 'barh', color = 'green', label='Wind Direction 3pm')
plt.title('Wind Directions')
plt.show()

#
# drop any missing index or values that are missing
#
dataset = dataset.dropna()

#
# convert/encode the Wind Direction Labels to numeric values the classifiers understand
#
label_encoder = LabelEncoder() 
dataset.WindGustDir = label_encoder.fit_transform(dataset.WindGustDir)
dataset.WindDir9am = label_encoder.fit_transform(dataset.WindDir9am)
dataset.WindDir3pm = label_encoder.fit_transform(dataset.WindDir3pm)

#
# Convert the Yes/No values for RainTomorrow labels and RainToday feature
# for the classifiers to understand
#
dataset['RainToday'] = dataset['RainToday'].apply(lambda x:1 if x == 'Yes' else 0)
dataset['RainTomorrow'] = dataset['RainTomorrow'].apply(lambda x:1 if x == 'Yes' else 0)
        

print("After Label Encodeing, check out the first ten rows of the" + 
        "first 12 columns with iloc[:, 0:12].head(10)")
print( dataset.iloc[:, 0:12].head(10) )

print("After Lable Encoding, the first ten rows of the last columns with iloc[:, 12:30].head(10)")
#
# iloc[:, 12:30] wont throw and exception, just dumps them all.
# probably cleaner to use 12:23 since we know the number of columns from the dataset.shape
# cleaner even still to use dataset.shape[1] to get the number of columns.
#
print( dataset.iloc[:, 12:dataset.shape[1]].head(10) )


#
# get the training data without the label 'RainTomorrow' and 'Date' columns
#
X = dataset.drop( ['Date', 'RainTomorrow'], axis='columns' )
y = dataset['RainTomorrow']

print("The columns for our testing data after dropping the date and label")
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

print("X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape))
print("X_test.shape: {}, y_test.shape: {}".format(X_test.shape, y_test.shape))

#
# train a Decision Tree classifier
#
import sklearn, sklearn.tree
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


#
# Gaussian Naive Bayes Classifier
#
from sklearn.naive_bayes import GaussianNB
gaussian_nb_classifier = GaussianNB()
gaussian_nb_classifier.fit(X_train, y_train)

print("Gaussian Naive Bayes Classifier: ")
print(gaussian_nb_classifier.get_params())

y_pred = gaussian_nb_classifier.predict(X_test)
gaussian_nb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Gaussian Naive Bayes Confusion Matrix:")
print(gaussian_nb_confusion_matrix)

#
# Finally calculate the accuracy, recall, and precision 
# for the Gaussian Naive Bayes Classifier
#
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
print( "Gaussian Naive Bayes Accuracy: {}, Recall: {}, Precision: {}".format(
    accuracy, recall, precision))

