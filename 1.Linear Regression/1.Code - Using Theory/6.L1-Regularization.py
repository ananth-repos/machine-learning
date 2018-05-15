# Demonstration of L1 regularization
# All equations are from theory notebook

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data.
    # 3. Calculating weights using gradient descent.
    # 4. Plot the costs.
    # 5. Plot our w vs true w.
    
# 1.Imports:
import numpy as np
import matplotlib.pyplot as plt

# 2.Generate sample data:
N = 50
D = 50

# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5)*10

# true weights - only the first 3 dimensions of X affect Y
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

# generate Y - add noise with variance 0.5
Y = X.dot(true_w) + np.random.randn(N)*0.5

# 3. Calculating weights using gradient descent:
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0 # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
for t in range(200):
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

  # find and store the cost
  mse = delta.dot(delta) / N
  costs.append(mse)

# 4.Plot the costs
plt.figure(1)
plt.plot(costs)
plt.xlabel('Iteration no.')
plt.ylabel('Cost')
plt.title('Change in cost')


# 5.Plot l1-w vs true w
plt.figure(2)
plt.plot(true_w, label='true weights')
plt.plot(w, label='L1-weights')
plt.xlabel('No.')
plt.ylabel('Weights')
plt.title('L1 Regularization weights vs. true weights')
plt.legend()
