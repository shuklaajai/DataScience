#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:08:54 2019

@author: ashukla
"""
############################################################################
#Import statements for necessary package(s).
import numpy as np
import pandas as pd
#################################################################
#Create a numeric numpy array, named arr1, with at least 30 items that contains outliers.
##########################################################################################
#We find the z score for each of the data point in the dataset and if the z score is greater than 3 than we can classify that point as an outlier. 
#Any point outside of 3 standard deviations would be an outlier.
arr1=np.array([10,12,12, 13,12,11,14,13,15,10,10, 10, 100,12, 14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10])
len(arr1)
outliers=[] 
################################################################################
#Create a numpy array that should have been numeric, named arr2. arr2 contains improper non-numeric missing values, like "?".
arr2=np.array([2, 1, " ", 1, 99, 1, 5, 3, "?", 1, 4, 3])
###############################################################################
#Create (define) a function, named “remove_outlier”, that removes outliers in arr1.
def remove_outliner(arr1):
 
    threshold=3
    mean_1 = np.mean(arr1)
    std_1 =np.std(arr1)
    
    
    for y in arr1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

#we now pass dataset that we created earlier and pass that as an input argument to the detect_outlier function
outlier_datapoints = remove_outliner(arr1)
print(outlier_datapoints)
#An outlier is a point which falls more than 1.5 times the interquartile range above the third quartile or below the first quartile.
#Fist sorting the dataset
sorted(arr1)
#Finding first quartile and third quartile
q1, q3= np.percentile(arr1,[25,75])
q1
q3
iqr = q3 - q1
#Find lower and upper bound
lower_bound = q1 - (1.5 * iqr) 
lower_bound

upper_bound = q3 + (1.5 * iqr)
upper_bound

FlagGood = (arr1 >= lower_bound) & (arr1 <= upper_bound)
FlagGood
arr1[FlagGood]
arr1 = arr1[FlagGood]
arr1
################################################################################# 
#Create (define) a function, named “replace_outlier”, that replaces outliers in arr1 with the arithmetic mean of the non-outliers.
############################################################################################################
def replace_outlier(arr1):
# calculate the limits for values that are not outliers.
    lower_bound = np.mean(arr1) + 2*np.std(arr1)
    upper_bound = np.mean(arr1) - 2*np.std(arr1)
    # Create Flag for values outside of limits
FlagBad = (arr1 < lower_bound) | (arr1 > upper_bound)
# present the flag
FlagBad
arr1[FlagBad] = np.mean(arr1)
# See the values of arr1
arr1
#######################

# FlagGood is the complement of FlagBad
FlagGood = ~FlagBad

# Replace outleiers with the mean of non-outliers
arr1[FlagBad] = np.mean(arr1[FlagGood])
# See the values of xarr1
arr1

################################################################################################
Create (define) a function, named “fill_median”, that fills in the missing values in arr2 with the median of arr2.

def fill_median(arr2):

   sum(arr2 > 4)

# Find out the data type for x:
type(arr2)

# Find out the data type for the elements in the array
arr2.dtype.name
#################

# Do not allow specific texts
FlagGood = (arr2 != "?") & (arr2 != " ")

# Find elements that are numbers
FlagGood = [element.isdigit() for element in arr2]
##################

# Select only the values that look like numbers
arr2 = arr2[FlagGood]

arr2
##################

# Need to cast the numbers from text (string) to real numeric values
arr2 = arr2.astype(int)

arr2
########################################################################################################
