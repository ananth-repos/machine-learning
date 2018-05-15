# This code shows how a linear regression analysis can be applied to a 2-dimensional data
# Implementation here is based on the theory described in the jupyter notebook.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data & save it as a csv file (Stored as a csv file just to use pandas).
    # 3. Load the dataset using pandas (X - inputs/feature, Y - output/target).
    # 4. Plot the generated data understand the trend.
    # 5. Calculate weights (parameters - a & b) using the equation from the theory lecture.
    # 6. Calculate Yhat from the weights above. Yhat = a*X + b.
    # 7. Calculate R-squared using the equation from the theory lecture to validate the model.
    
# 1.Imports:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 2.Generate sample data:
N = 100
w = np.array([2, 3])
with open('data_2d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=(N,2))
    Y = np.dot(X, w) + 1 + np.random.normal(scale=5, size=N)
    for i in range(N):
        f.write("%s,%s,%s\n" % (X[i,0], X[i,1], Y[i]))
        
# 3.Load the data:
df = pd.read_csv('data_2d.csv',header = None)
df['ones'] = np.ones(len(X))
X = df[[0,1,'ones']].as_matrix()
Y = df[2].values

# 4.Plot the data:
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# 5.Model: Y = a*X + B
# Apply the equations from the jupyter notebook to calculate a & b:
# Denominator is same for both a & b
w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

# 6.Predict Y:
Yhat = np.dot(X,w)

# 7.R-squared:
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)

print('the r-squared is {}'.format(r2))
