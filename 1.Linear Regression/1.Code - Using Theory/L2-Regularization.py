# Demonstration of L2 regularization
# Also comparison between maximum likelihood estimation(MLE) & maxiumum a posteriori solution(MAP) is done
# All equations are from theory notebook

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Generate sample data with some outliers.
    # 3. Plot the generated data.
    # 4. Add bias term.
    # 5. Plot the MLE solution.
    # 6. Plot the L2-Reg solution & compare it with MLE.

# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt

# 2. Generate sample data:
N = 50
X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

# make outliers
Y[-1] += 30
Y[-2] += 30

# 3.Plot the data
plt.scatter(X, Y)
plt.show()

# 4.Add bias term
X = np.vstack([np.ones(N), X]).T

# 5. Plot the MLE solution.
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.figure(1)
plt.scatter(X[:,1], Y,label = 'data',c = 'green')
plt.plot(X[:,1], Yhat_ml,label = 'MLE',c = 'red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('MLE Solution')


# 6. Plot the regularized solution
# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.figure(2)
plt.scatter(X[:,1], Y,label = 'data',c = 'green')
plt.plot(X[:,1], Yhat_ml, label='MLE',c = 'red')
plt.plot(X[:,1], Yhat_map, label='MAP', c = 'black')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('MLE vs. MAP')