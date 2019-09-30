#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 02:00:09 2019

@author: ashukla
"""
#Import statements for necessary package(s).
#############################################################################################################
import pandas as pd
import numpy as np
#####################################################################

#Read in the dataset from a freely and easily available source on the internet.
######################################################################################################################
# Origin of data:
# https://archive.ics.uci.edu/ml/datasets/car+evaluation
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None) 
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
# Check the data types
Mamm.dtypes
# Check the first rows of the data frame
Mamm.head()
# Check the unique values
Mamm.loc[:, "Shape"].unique()
#######################################################################################################################

#Normalize numeric values (at least 1 column, but be consistent with other numeric data).
#####################################################################################################################
# Normalizing a numpy array

# First we need an array that we can normalize
# We call this array a variable
x = np.array([1,11,5,3,15,3,5,9,7,9,3,5,7,3,5,21])
print(" The original variable:", x)
############

# Trivial Normalization
#   offset is 0
#   spread is 1
# In math a trivial process is one that doesn't change anything
offset = 0
spread = 1
xNormTrivial = (x - offset)/spread
print(" Trivial normalization doesn't change values: ", xNormTrivial)
########################################################################################################################

#Bin numeric variables (at least 1 column).
####################################################################################################################
# Equal-width Binning
# Determine the boundaries of the bins
NumberOfBins = 3
BinWidth = (max(x) - min(x))/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = min(x) + 1 * BinWidth
MaxBin2 = min(x) + 2 * BinWidth
MaxBin3 = float('inf')

print(" Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print(" Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print(" Bin 3 is greater than", MaxBin2, "up to", MaxBin3)
#############################################################################################################
#Decode categorical data.
Replace = Mamm.loc[:, "Shape"] == "1"
Mamm.loc[Replace, "Shape"] = "round"

Replace = Mamm.loc[:, "Shape"] == "2"
Mamm.loc[Replace, "Shape"] = "oval"

Replace = Mamm.loc[:, "Shape"] == "3"
Mamm.loc[Replace, "Shape"] = "lobular"

Replace = Mamm.loc[:, "Shape"] == "4"
Mamm.loc[Replace, "Shape"] = "irregular"
#############################################################################################################
#Impute missing categories.
########################################################################################################
# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
# Specify all the locations that have a missing value
MissingValue = Mamm.loc[:, "Shape"] == "?"
# Impute missing values
Mamm.loc[MissingValue, "Shape"] = "irregular"
###################################################################################################
#Consolidate categorical data (at least 1 column).
#Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"
################################################################################################
#One-hot encode categorical data with at least 3 categories (at least 1 column).
#Many times we will have our data in a pandas data frame. Pandas has built in functionality to help us perform one hot encoding
#First let’s generate our test data using the code below
import pandas as pd
import numpy as np

A = np.linspace(2.0, 10.0, num=3)
B = ['dog','cat','bird']
d = {'numeric': A, 'categorical': B}

df = pd.DataFrame(d)
#We now have a pandas data frame df as shown in the below image with a categorical variable column and a numerical one.
#We will now convert our categorical variable into its one hot encoding representation. To do this, first we cast our categorical variable into the built in pandas Categorical data type.
df['categorical'] = pd.Categorical(df['categorical'])
dfDummies = pd.get_dummies(df['categorical'], prefix = 'category')
#To add these 3 columns to our original data frame we can use the concat function as below
df = pd.concat([df, dfDummies], axis=1)
df
##Conclusion is One hot encoding is a powerful technique to transform categorical data into a numerical representation that machine learning algorithms can utilize to perform optimimally without falling into the misrepresentation 
###########################################################################################
#Remove obsolete columns.
Mamm = Mamm.drop("Shape", axis=1)
# Create 3 new columns, one for each state in "Margin"
Mamm.loc[:, "ill-d"] = (Mamm.loc[:, "Margin"] == "ill-defined").astype(int)
Mamm.loc[:, "circu"] = (Mamm.loc[:, "Margin"] == "circumscribed").astype(int)
Mamm.loc[:, "spicu"] = (Mamm.loc[:, "Margin"] == "spiculated").astype(int)
# Remove obsolete column
Mamm = Mamm.drop("Margin", axis=1)
####################################################################################################
#Present plots for 1 or 2 categorical columns.
# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
#########################################################################################################
