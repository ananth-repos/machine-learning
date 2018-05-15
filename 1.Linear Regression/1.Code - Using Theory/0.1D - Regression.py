# This code shows how a linear regression analysis can be applied to a 1-dimensional data
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
    # 9. How is the fit? Good? Bad? Ok?

# 1.Imports:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2.Generate sample data:
N = 100
with open('data_1d.csv', 'w') as f:
    X = np.random.uniform(low=0, high=100, size=N)
    Y = 2*X + 1 + np.random.normal(scale=5, size=N)
    for i in range(N):
        f.write("%s,%s\n" % (X[i], Y[i]))
        
# 3.Load the data:
df = pd.read_csv('data_1D.csv',header = None)
X = df[0].values
Y = df[1].values

# 4.Plot the data:
plt.figure(1)
plt.scatter(X,Y,c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data Scatter Plot')


# 5.Model: Y = a*X + B
# Apply the equations from the jupyter notebook to calculate a & b:
# Denominator is same for both a & b
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denominator
b = (Y.mean()*X.dot(X) - X.mean() * X.dot(Y))/denominator

# 6.Predict Y:
Yhat = a*X + b

# 7.Plot predicted vs actual:
plt.figure(2)
plt.scatter(X,Y,c = 'red',label = 'Actual Data')
plt.plot(X,Yhat,c = 'black',label = 'Predicted Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Actual vs. Model')
plt.legend()


# 8 & 9.R-squared:
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)
print('the r-squared is {} & hence the model is good'.format(r2))
