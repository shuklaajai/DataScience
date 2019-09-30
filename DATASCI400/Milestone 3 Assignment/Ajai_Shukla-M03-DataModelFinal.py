#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:24:59 2019

@author: ashukla
"""
"""
The goal is to create a model that performs the best, where “best” is defined by:
•Data: the historical data that you have available.
•Time: the time you have to spend on the project.
•Procedure: the data preparation steps, algorithm or algorithms, and the chosen algorithm configurations.
The model is the pinnacle of this process, and in this model we will be able to  making predictions.
The Purpose of Train/Test Sets:
Why do we use train and test sets?
Creating a train and test split of your dataset is one method to quickly evaluate the performance of an algorithm on your problem. The training dataset is used to prepare a model, to train it. Comparing the predictions and withheld outputs on the test dataset allows us to compute a performance measure for the model on the test dataset. This is an estimate of the skill of the algorithm trained on the problem when making predictions on unseen data.
When we evaluate an algorithm, we are in fact evaluating all steps in the procedure, including how the training data was prepared (e.g. scaling), the choice of algorithm (e.g. kNN), and how the chosen algorithm was configured (e.g. k=3).
The performance measure calculated on the predictions is an estimate of the skill of the whole procedure.
We generalize the performance measure from:
•“the skill of the procedure on the test set “
to
•“the skill of the procedure on unseen data “.
This is quite a leap and requires that:
•The procedure is sufficiently robust that the estimate of skill is close to what we actually expect on unseen data.
•The choice of performance measure accurately captures what we are interested in measuring in predictions on unseen data.
•The choice of data preparation is well understood and repeatable on new data, and reversible if predictions need to be returned to their original scale or related to the original input values.
In fact, using the train/test method of estimating the skill of the procedure on unseen data often has a high variance (unless we have a heck of a lot of data to split). This means that when it is repeated, it gives different results, often very different results.
The outcome is that we may be quite uncertain about how well the procedure actually performs on unseen data and how one procedure compares to another.
Often, time permitting, we prefer to use k-fold cross-validation instead.
The Purpose of k-fold Cross Validation
Why do we use k-fold cross validation?
Cross-validation is another method to estimate the skill of a method on unseen data. Like using a train-test split.
Cross-validation systematically creates and evaluates multiple models on multiple subsets of the dataset.
We pretend the test dataset is new data where the output values are withheld from the algorithm. We gather predictions from the trained model on the inputs from the test dataset and compare them to the withheld output values of the test set.
Comparing the predictions and withheld outputs on the test dataset allows us to compute a performance measure for the model on the test dataset. This is an estimate of the skill of the algorithm trained on the problem when making predictions on unseen data.
Let’s unpack this further
When we evaluate an algorithm, we are in fact evaluating all steps in the procedure, including how the training data was prepared (e.g. scaling), the choice of algorithm (e.g. kNN), and how the chosen algorithm was configured (e.g. k=3).
The performance measure calculated on the predictions is an estimate of the skill of the whole procedure.
We generalize the performance measure from:
•“the skill of the procedure on the test set“
"""
#We are going to use the iris flowers dataset. This dataset is famous because it is used as the “hello world” dataset in machine learning and statistics by pretty much everyone.
#The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.
#2.1 Import libraries
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#3. Summarize the Dataset,3.1 Dimensions of Dataset
#We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
# shape
print(dataset.shape)
#3.2 Peek at the Data
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())
#4. Data Visualization
#4.1 Univariate Plots
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()
# histograms
dataset.hist()
plt.show()
#4.2 Multivariate Plots

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
#5. Evaluate Some Algorithms
#Here is what we are going to cover in this step:

#Separate out a validation dataset.
#Set-up the test harness to use 10-fold cross validation.
#Build 5 different models to predict species from flower measurements
#Select the best model.
#We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
# Split-out validation dataset
#5.1 Create a Validation Dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#You now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation
#5.2 Test Harness
#We will use 10-fold cross validation to estimate accuracy.This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#5.3 Build Models
#Let’s evaluate 6 different algorithms:

#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes (NB).
#Support Vector Machines (SVM).
#Let’s build and evaluate our models:
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
#5.4 Select Best Model
#We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.
#Running the example above, we get the following raw results:
# LR: 0.966667 (0.040825)
#LDA: 0.975000 (0.038188)
#KNN: 0.983333 (0.033333)
#CART: 0.966667 (0.040825)
#NB: 0.975000 (0.053359)
#SVM: 0.991667 (0.025000)
    ### In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score.
 ##We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.
#####################################################
####6. Make Predictions
#######################################################
##The KNN algorithm is very simple and was an accurate model based on our tests. Now we want to get an idea of the accuracy of the model on our validation set.
#We will run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
#We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. 
#Finally, the classification report provides a breakdown of each class by precision, recall