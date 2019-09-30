#Write a script to prepare a data set with known missing values.  The data could be the same data that you will use for a predictive model 
#in Milestone 3.  The script must run, without alterations, in the grader's environment.


""" 
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################

 """
#####################################################################################################################
#Read in the data from a freely available source on the internet.  
#####################################################################################################################
 
import urllib.request as ur 
import json
url = "https://www.govtrack.us/data/congress/113/votes/2013/s11/data.json"
data = json.load(ur.urlopen(url))
print(data)

#####################################################################################################################
#Account for outlier values in numeric columns (at least 1 column).
###################################################################################################################
# Import the Numpy library
import numpy as np
# Create an array with the data
x = np.array([2, 1, 1, 99, 1, 5, 3, 1, 4, 3])
# The high limit for acceptable values is the mean plus 2 standard deviations    
LimitHi = np.mean(x) + 2*np.std(x)
# The high limit is the cutoff for good values
LimitHi
# The low limit for acceptable values is the mean plus 2 standard deviations
LimitLo = np.mean(x) - 2*np.std(x)
# The low limit is the cutoff for good values
LimitLo
# Flag for outliner values within limits 
FlagGood = (x >= LimitLo) & (x <= LimitHi)
# What type of variable is FlagGood? Check the Variable explorer.
# present the flag
FlagGood
# We can present the values of the items within the limits
x[FlagGood]
# Overwrite x with the selected values
x = x[FlagGood]
# present the data set
x
#####################################################################################################################
#Replace missing numeric data (at least 1 column).
#####################################################################################################################
import numpy as np
# Create an array with the data
x = np.array([2, 1, 1, 99., 1, 5, 3, 1, 4, 3])
# calculate the limits for values that are not outliers. 
LimitHi = np.mean(x) + 2*np.std(x)
LimitLo = np.mean(x) - 2*np.std(x)
# Create Flag for values outside of limits
FlagBad = (x < LimitLo) | (x > LimitHi)
# present the flag
FlagBad
# Replace outlieres with mean of the whole array
x[FlagBad] = np.mean(x)
# See the values of x
x
# FlagGood is the complement of FlagBad
FlagGood = ~FlagBad
# Replace outleiers with the mean of non-outliers
x[FlagBad] = np.mean(x[FlagGood])
# See the values of x
x
# Get the Sample data
x = np.array([2, 1, 1, 99., 1, 5, 3, 1, 4, 3])
# Replace outliers with the median of the whole array
x[FlagBad] = np.median(x)
# See the values of x
x
#####################################################################################################################
#Normalize numeric values (at least 1 column, but be consistent with numeric data).
#####################################################################################################################
import numpy as np
# Normalizing a numpy array
# First we need an array that we can normalize
# We call this array a variable
x = np.array([1,11,5,3,15,3,5,9,7,9,3,5,7,3,5,21])
print(" The original variable:", x)
# Trivial Normalization
#   offset is 0
#   spread is 1
# In math a trivial process is one that doesn't change anything
offset = 0
spread = 1
xNormTrivial = (x - offset)/spread
print(" Trivial normalization doesn't change values: ", xNormTrivial)
#####################################################################################################################
#Bin numeric variables (at least 1 column).
#####################################################################################################################
import numpy as np
# Variable
x = np.array((81,3,3,4,4,5,5,5,5,5,5,5,5,5,5,6,6,7,7,7,7,8,8,9,12,24,24,25))
# We want a categorical array instead of the above numeric variable
# The categories in the categorical variable should be "Low", "Med", and "High" 
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
#####################################################################################################################
#Consolidate categorical data (at least 1 column).
#####################################################################################################################
import pandas as pd
# Download the data
# http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
# The category columns are decoded and missing values are imputed
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "round"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "microlobulated"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "obscured"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
# Check the first rows of the data frame
Mamm.head()
# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
Mamm.loc[:,"Margin"].value_counts()
# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
# Simplify Shape by consolidating oval and round
Mamm.loc[Mamm.loc[:, "Shape"] == "round", "Shape"] = "oval"
# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
# Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"
# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
#####################################################################################################################
#One-hot encode categorical data with at least 3 categories (at least 1 column).
#####################################################################################################################
import pandas as pd
# Download the data
# http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
# The category columns are decoded and missing values are imputed
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "round"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "microlobulated"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "obscured"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
# Check the first rows of the data frame
Mamm.head()
# Get the counts for each value
Mamm.loc[:,"Shape"].value_counts()
Mamm.loc[:,"Margin"].value_counts()
# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
# Simplify Shape by consolidating oval and round
Mamm.loc[Mamm.loc[:, "Shape"] == "round", "Shape"] = "oval"
# Plot the counts for each category
Mamm.loc[:,"Shape"].value_counts().plot(kind='bar')
# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
# Simplify Margin by consolidating ill-defined, microlobulated, and obscured
Mamm.loc[Mamm.loc[:, "Margin"] == "microlobulated", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "obscured", "Margin"] = "ill-defined"
# Plot the counts for each category
Mamm.loc[:,"Margin"].value_counts().plot(kind='bar')
#####################################################################################################################
#Remove obsolete columns.
#####################################################################################################################
import pandas as pd
# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
# Create 3 new columns, one for each state in "Shape"
Mamm.loc[:, "oval"] = (Mamm.loc[:, "Shape"] == "oval").astype(int)
Mamm.loc[:, "lobul"] = (Mamm.loc[:, "Shape"] == "lobular").astype(int)
Mamm.loc[:, "irreg"] = (Mamm.loc[:, "Shape"] == "irregular").astype(int)
# Remove obsolete column
Mamm = Mamm.drop("Shape", axis=1)
# Create 3 new columns, one for each state in "Margin"
Mamm.loc[:, "ill-d"] = (Mamm.loc[:, "Margin"] == "ill-defined").astype(int)
Mamm.loc[:, "circu"] = (Mamm.loc[:, "Margin"] == "circumscribed").astype(int)
Mamm.loc[:, "spicu"] = (Mamm.loc[:, "Margin"] == "spiculated").astype(int)
# Remove obsolete column
Mamm = Mamm.drop("Margin", axis=1)
# Check the first rows of the data frame
Mamm.head()
# Check the data types
Mamm.dtypes