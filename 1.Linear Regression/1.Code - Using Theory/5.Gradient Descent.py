# Gradient Descent Implementation (fromt theory notebook)
# Demonstrated multi-collinearity problem from artificially generated dataset

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data.
    # 3. Calculate weights using gradient descent.
    # 4. Plot the costs.
    # 5. Plot prediction vs actual.
    
# 1. Imports:
import numpy as np
import matplotlib.pyplot as plt

# 2. Generate sample data:
N = 10
D = 3
X = np.zeros((N, D))
X[:,0] = 1 # bias term
X[:5,1] = 1
X[5:,2] = 1
Y = np.array([0]*5 + [1]*5)

# 3. Calculate weights using gradient descent:
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# won't work because of multicollinearity
# Hence gradient descent
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001 

for t in range(1000): # 1000 iterations chosen
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*X.T.dot(delta)

  # find and store the cost
  mse = delta.dot(delta) / N # mean-sq error
  costs.append(mse) # append cost

# 4.Plot the costs:
plt.figure(1)
plt.plot(costs)
plt.xlabel('Iteration no.')
plt.ylabel('Cost')
plt.title('Change in cost')

# 5.Plot prediction vs actual
plt.figure(2)
plt.plot(Yhat, label='Prediction')
plt.plot(Y, label='Actual')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Actual vs Prediction - Gradient Descent')
plt.legend()
plt.show()