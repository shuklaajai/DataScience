#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:00:23 2019

@author: ashukla
"""
###################################################################
#Import statements for necessary package(s).
import pandas as pd
#####################################################################
#Read in the dataset from a freely and easily available source on the internet.
# Origin of data:
# https://archive.ics.uci.edu/ml/datasets/car+evaluation
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
Auto = pd.read_csv(url,sep='\s+', header=None)
#################################################################################
#Assign reasonable column names.
# https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names
Auto.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
               'acceleration', 'model year', 'origin', 'car name']
# Get size of data set
Auto.shape
# View first few rows
Auto.head()
# Present values from the column
Auto.loc[:,'mpg']
# Present sorted values
Auto.loc[:,'mpg'].sort_values()
##################################################################################################################
#Impute and assign median values for missing numeric values.Replace outliers.
# The url for the data
##################################################################################################################
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
# Download the data
Mamm = pd.read_csv(url, header=None)
# Check the first rows of the data frame
Mamm.head()
# Replace the default column names 
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
# Check the first rows of the data frame
Mamm.head()
# Check data types of columns
Mamm.dtypes
# Check distinct values 
Mamm.loc[:,"BI-RADS"].unique()
# Convert BI-RADS to numeric data including nans
Mamm.loc[:, "BI-RADS"] = pd.to_numeric(Mamm.loc[:, "BI-RADS"], errors='coerce')
# Check the data types of the columns in the data frame
Mamm.dtypes
# Check distinct values 
Mamm.loc[:,"BI-RADS"].unique()
# Calculation on a column with nan values
np.median(Mamm.loc[:,"BI-RADS"])
# Determine the elements in the "BI-RADS" column that have nan values
HasNan = np.isnan(Mamm.loc[:,"BI-RADS"])
# Determine the median from a column with NANs
np.median(Mamm.loc[~HasNan,"BI-RADS"])
# Determine the median from a column with NANs
np.nanmedian(Mamm.loc[:,"BI-RADS"])
# The replacement value for NaNs is Median
Median = np.nanmedian(Mamm.loc[:,"BI-RADS"])
# Median imputation of nans
Mamm.loc[HasNan, "BI-RADS"] = Median
############################################################################################################
#Create a histogram of a numeric variable. Use plt.show() after each histogram.
# Make Histogram of miles per gallon
import matplotlib.pyplot as plt
plt.hist(Auto.loc[:,'mpg'])
###############################################################################################################
#Create a scatterplot. Use plt.show() after the scatterplot.
# plot type is the simple scatter plot, Instead of points being joined by line segments, here the points are represented individually with a dot, circle, or other shape. We’ll start by setting up the notebook for plotting and importing the functions we will use
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#Scatter Plots with plt.plot
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black');
#argument in the function call is a character that represents the type of symbol used for the plotting
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);
#these character codes can be used together with line and color codes to plot points along with a line connecting them
plt.plot(x, y, '-ok');
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);
#Scatter Plots with plt.scatter
plt.scatter(x, y, marker='o');
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();  # show color scale
plt.show()

###########################################################################################
#Determine the standard deviation of all numeric variables. Use print() for each standard deviation.
