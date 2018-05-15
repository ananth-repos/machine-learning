# Objective is to predict blood pressure from age & weight. 
# R-sq values are calcualted for all combination of inputs (age only, weight only & both)
# Data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X0 = systolic blood pressure
# X1 = age in years
# X2 = weight in pounds

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Load the dataset using pandas (X - input/feature, Y - output/target).
    # 3. Plot the generated data understand the trend.
    # 4. Create input features for all combinations.
    # 5. get_r2 function defintion that calculates weights, predictions & final r-sq values.
    # 6. Print results for all cases.
    
# 1. Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 2. Load dataset:
df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# 3. Plot the data to understand the trends:
plt.figure(1)
plt.scatter(X[:,1],X[:,0])
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Blood pressure vs. Age')

plt.figure(2)
plt.scatter(X[:,2],X[:,0])
plt.xlabel('Weight (pounds)')
plt.ylabel('Systolic Blood Pressure')
plt.title('Blood pressure vs. Weight (pounds)')

# 4. Create input features for all combinations:
df['ones'] = 1

Y = df['X1']
X = df[['X2','X3','ones']]
X2only = df[['X2','ones']]
X3only = df[['X3','ones']]

# 5. Function definition that calcualtes weights, predictions & R-sq values:
def get_r2(X,Y):
    w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
    Yhat = np.dot(X,w)
    # R-sq:
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)

    return r2

# 6. Print results:    
print('the r-squared for x2 only is:',get_r2(X2only,Y))
print('the r-squared for x3 only is:',get_r2(X3only,Y))
print('the r-squared for x2 & x3 only is:',get_r2(X,Y))