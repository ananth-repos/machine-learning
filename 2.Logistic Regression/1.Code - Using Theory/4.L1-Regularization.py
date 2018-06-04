# Gradient descent with L1-Regularization.
# Section 5 of the Jupyter Notebook for derivation/theory.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data from std. normal distribution.
    # 3. True weights - only the first 3 dimensions of X affect Y.
    # 4. Define sigmoid function.
    # 5. Calculate true Y.
    # 6. Plot the costs - Figure 1.
    # 7. Plot L1 - w vs. true w - Figure 2.
    
# 1. Import all relevante libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Create input data:
N = 50 # No. of samples
D = 50 # No. of features

# Uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5)*10

# 3. True weights - only the first 3 dimensions of X affect Y:
true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

# 4. Sigmoid function
def sigmoid(z):
  return 1/(1 + np.exp(-z))

# 5. Calculate true Y:
# add some noise with variance 0.5
Y = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

# 6. Gradient Descent with L1 Regularization:
# Section 5 of the Jupyter Notebook
costs = [] # keep track of cross-entropy error
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001 # trial & error
l1 = 3.0 # lambda 
for t in range(400):
  # update w using l1 MAP:
  Yhat = sigmoid(X.dot(w)) # prediction
  delta = Yhat - Y
  w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w)) # gradient descent 

  # Find and store the cost
  cost = -(Y*np.log(Yhat) + (1-Y)*np.log(1 - Yhat)).mean() + l1*np.abs(w).mean()
  costs.append(cost)

# 6. Plot the costs:
plt.figure(1)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Convergence via Gradient Descent')

# 7. Plot L1 - w vs. true w:
plt.figure(2)
plt.plot(true_w, label='true weight')
plt.plot(w, label='L1 weights(MAP)')
plt.xlabel('weights 0-49 (total 50)')
plt.ylabel('Value of the each weight')
plt.title('True weights vs L1 weights')
plt.legend()
plt.show()

print('Notice how the weights other than first 3 are close to zero!')