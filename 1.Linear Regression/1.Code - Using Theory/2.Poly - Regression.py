# This code shows how a linear regression analysis can be applied to a poly dimensional data
# Implementation here is based on the theory described in the jupyter notebook.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data & save it as a csv file (Stored as a csv file just to use pandas).
    # 3. Load the dataset using pandas (X - input/feature, Y - output/target).
    # 4. Plot the generated data understand the trend.
    # 5. Calculate weights (parameters - a & b) using the equation from the theory lecture.
    # 6. Calculate Yhat from the weights above. Yhat = a*X + b.
    # 7. Plot actual vs. predicted to visualize the fit.
    # 8. Calculate R-squared using the equation from the theory lecture to validate the model.

# 1.Imports:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 2.Generate sample data:
N = 100
with open('data_poly.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N)
    X2 = X*X
    Y = 0.1*X2 + X + 3 + np.random.normal(scale=10, size=N)
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))

# 3.Load the data:
df = pd.read_csv('data_poly.csv',header = None)
X1 = df[0].values
X2 = X1*X1
X3 = np.ones(len(X1))
Y = df[1].values
X = np.vstack((X1,X2,X3)).transpose()

# 4.Plot the data:
fig = plt.figure(1)
plt.scatter(X[:,0],Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 5.Model: Y = a*X + B
# Apply the equations from the jupyter notebook to calculate a & b:
# Denominator is same for both a & b
w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

# 6.Predict Y:
Yhat = np.dot(X,w)

# 7.Plot predicted vs actual:
plt.scatter(X[:,0],Y,c='red',label = 'Actual Data')
plt.plot(sorted(X[:,0]),sorted(Yhat),c='blue',label = 'Predicted Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Actual vs. Model')
plt.show()

# 8.R-squared:
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print('the r-squared is {}'.format(r2))