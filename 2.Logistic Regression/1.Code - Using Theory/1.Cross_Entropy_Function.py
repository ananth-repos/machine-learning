# Implementation of cross-entropy error function using numpy.
# Compare the random weight results with the closed form solution (0,4,4).

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data from std. normal distribution.
    # 3. Create random weights for plotting.
    # 4. Define sigmoid function.
    # 5. Feedforward calculation.
    # 6. Cross-entropy error function.
    # 7. Print cross-entropy error from random weights.
    # 8. Closed-form solution.
    # 9. Print cross-entropy error from closed form solution.
    
# 1. Import all relevante libraries:
import numpy as np

# 2. Create input data:
N = 100 # No. of samples
D = 2 # No. of features

# Random data generation:
X = np.random.randn(N,D)

# Center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# Center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# Labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50) # Random labels

# Bias term: Add a column of ones
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# 3. Create random weights:
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

# 7. Print cross-entropy error from random weights:
print(cross_entropy(T, Y))

# 8. Closed-form solution:
# Section 3A from jupyter notebook
w = np.array([0, 4, 4])

# calculate the model output
z = Xb.dot(w)
Y = sigmoid(z)

# 9. Print cross-entropy error from closed form solution:
print(cross_entropy(T, Y))
print('As we expect the error is much lower with our closed form solution!')
