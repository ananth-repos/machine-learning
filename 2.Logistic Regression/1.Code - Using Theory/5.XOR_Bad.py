# XOR problem using logistic regression

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data.
    # 3. Generate random weights.
    # 4. Define sigmoid function.
    # 5. Calculate Y.
    # 6. Define Cross-entropy error function.
    # 7. Gradient Descent.
    # 8. Plots - Error & Weights-Mag.
    # 9. Classification Rate.

# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Create input data:
N = 4 # No. of samples
D = 2 # No. of features

# XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# Target:
T = np.array([0, 1, 1, 0])

# Display the data:
plt.figure(1)
plt.scatter(X[:,0], X[:,1], s=100, c=T, alpha=0.5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('XOR Data')

# Bias term: Add a column of ones
ones = np.ones((N, 1))

# Concatenate X & bias term together:
Xb = np.concatenate((ones, X), axis=1)

# 3. Generate random weights:
w = np.random.randn(D + 1)

# 4. Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 5. Calculate Y:
z = Xb.dot(w)
Y = sigmoid(z)

# 6. Cross-entropy error function:
def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# 7. Gradient Descent:
# Section 3D of the Jupyter Notebook for derivation/theory.
    
learning_rate = 0.001 # trial & error
error = [] # keep track of cross-entropy error
w_mags = [] # keep track of magnitude of weights

for i in range(20000):
    e = cross_entropy(T, Y)
    error.append(e) # append error
    if i % 1000 == 0:
        # print error every 10 epochs
        print(e)

    # Gradient descent weight update:
    w += learning_rate * Xb.T.dot(T - Y)

    w_mags.append(w.dot(w)) # append w_mag

    # Calculate Y:
    Y = sigmoid(Xb.dot(w))
    
print('Final w:', w)

# 8. Plots - Error & Weights-Mag:
plt.figure(2)
plt.plot(error)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Error')
plt.title("Cross-Entropy Error vs. Iteration")

plt.figure(3)
plt.plot(w_mags)
plt.xlabel('Iteration')
plt.ylabel('Magnitude of weights')
plt.title('w - magnitudes')

# 9. Classification Rate:
# This tells us what % of values are classified correctly.
# We round the output of sigmoid to get either 0 or 1.
# Then we calcualte the sum of the difference between T and round(Y).
# Divide that by N
# Subtract from 1 to get classification rate.
print('Final classification rate:', 1 - np.abs(T - np.round(Y)).sum() / N) 
print('As you can see the classification rate in bad!')

