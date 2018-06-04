# Feedforward of the logistic unit using numpy.
# Data X and weight w are randomly generated from standard normal dist.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data from std. normal distribution.
    # 3. Create random weights for plotting.
    # 4. Define sigmoid function.
    # 5. Feedforward calculation.
    # 6. Plot the sigmoid input/output.
    
# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Create input data:
N = 100 # No. of samples
D = 2 # No. of features

# Random data generation:
X = np.random.randn(N,D)
ones = np.ones((N, 1)) # bias
Xb = np.concatenate((ones, X), axis=1) # Concatenate X & bias term

# 3. Create random weights:
w = np.random.randn(D + 1) # D+1 to account for bias term

# 4. Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 5. Feedforward:
z = Xb.dot(w)
y = sigmoid(z)
fig = plt.figure()

# 6. Plots:
plt.subplot(121)
plt.scatter(X[:,1],z,c = 'red',label = 'z')
plt.xlabel('Feature 1 X[:,0]')
plt.ylabel('z')
plt.legend()
plt.title('Input to Sigmoid')

plt.subplot(122)
plt.scatter(X[:,1],y, c = 'blue',label = 'y')
plt.xlabel('Feature 1 X[:,0]')
plt.ylabel('y')
plt.title('Output from Sigmoid')
plt.legend()
plt.show()

print('Notice how the output from sigmoid is always between 0-1!')

