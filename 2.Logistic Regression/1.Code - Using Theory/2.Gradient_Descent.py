# Implementation of gradient descent for logistic regression.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data from std. normal distribution.
    # 3. Create random weights for plotting.
    # 4. Define sigmoid function.
    # 5. Feedforward calculation.
    # 6. Cross-entropy error function.
    # 7. Gradient Descent.
    # 8. Plot the data/decision line.
    
# 1. Import all relevante libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Create input data:
N = 100 # No. of samples
D = 2 # No. of features

N_per_class = N//2 # Splitting data points equally between 2 classes.

# Random data generation:
X = np.random.randn(N,D)

# Center the first 50 points at (-2,-2)
X[:N_per_class,:] = X[:N_per_class,:] - 2*np.ones((N_per_class,D))

# Center the last 50 points at (2, 2)
X[N_per_class:,:] = X[N_per_class:,:] + 2*np.ones((N_per_class,D))

# Labels: first 50 are 0, last 50 are 1
T = np.array([0]*N_per_class + [1]*N_per_class)

# Bias term: Add a column of ones
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# 3. Generate random weights:
w = np.random.randn(D + 1)

# 4. Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 5. Feedforward:
z = Xb.dot(w)
Y = sigmoid(z)

# 6. Cross-entropy error function:
# Section 3B from jupyter notebook
def cross_entropy(T, Y):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

# 7. Gradient Descent:
# Section 3D from jupyter notebook
# let's do gradient descent 100 times
learning_rate = 0.1
for i in range(100):
    # print error every 10 epochs
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    # Gradient descent weight update:
    w += learning_rate * Xb.T.dot(T - Y)

    # Calculate Y:
    Y = sigmoid(Xb.dot(w))
    
print('Notice how the errors are converging!')

# 8. Plot the data/decision line:
plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha=0.5, label = 'Data') # plotting 2 Gaussian clouds
x_axis = np.linspace(-6, 6, 100)
y_axis = -(w[0] + x_axis*w[1]) / w[2] # Calculate y using the equation of a line ax+by+c = 0
plt.plot(x_axis, y_axis, label = 'Decision line') # plotting the decision boundary
plt.legend()
plt.title('Decision line using Logistic Regression')
plt.show()

