# Donut problem using logistic regression

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data.
    # 3. Plot the data.
    # 4. Add bias term.
    # 5. Add radius as a feature.
    # 6. Generate random weights for initialization.
    # 7. Define sigmoid function.
    # 8. Calculate Y.
    # 9. Define Cross-entropy error function.
    # 10. Gradient Descent with L2.
    # 11. Plot - Cross-entropy error.
    # 12. Classification Rate.
    
# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Generate input data:
N = 1000 # No. of samples
D = 2 # No. of features

# Create 2 clouds of data at different radii:
R_inner = 5 # inner radius
R_outer = 10 # outer radius

# Spread the data along the circumference of each clouds.
# Distance from origin is radius + random normal
# Angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
T = np.array([0]*(N//2) + [1]*(N//2)) # Labels: first 50 are 0, last 50 are 1

# 3. Plot the data:
plt.figure(1)
plt.scatter(X[:,0], X[:,1], c=T)
plt.xlabel('Cosine angles')
plt.ylabel('Sine angles')
plt.title('Donut Data')

# 4. Add bias term: 
ones = np.ones((N, 1))

# 5. Add radius as a feature: 
# a column of r = sqrt(x^2 + y^2)
r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
Xb = np.concatenate((ones, r, X), axis=1)

# 6. Initialize weights:
w = np.random.randn(D + 2)

# 7. Sigmoid Function:
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# 8. Calculate model output Y:
z = Xb.dot(w)
Y = sigmoid(z)

# 9. Cross-entropy function:
def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# 10. Gradient Descent with L2:
learning_rate = 0.0001 # trial & error
error = [] # keep track of cross-entropy error
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e) # append error
    if i % 500 == 0:
        print(e) # print error every 10 epochs

    # Gradient descent weight update with L2:
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w )

    # Calculate Y:
    Y = sigmoid(Xb.dot(w))

# 11. Plots:
plt.figure(2)
plt.plot(error)
plt.xlabel('Iteration')
plt.ylabel('Cross-Entropy Error')
plt.title("Cross-Entropy Error vs. Iteration")

# 12. Classification Rate:
# This tells us what % of values are classified correctly.
# We round the output of sigmoid to get either 0 or 1.
# Then we calcualte the sum of the difference between T and round(Y).
# Divide that by N
# Subtract from 1 to get classification rate.
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
