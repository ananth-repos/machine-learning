# Using KNN for the XOR problem.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Import KNN function from vectorized implementation.
    # 3. Main function.
        # 3.1 Generate XOR dataset.
        # 3.2 Display the data.
        # 3.3 KNN Fit & Predict.
        
# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt
import imp

# 2. Import KNN function from vectorized implementation:
with open('1.KNN_Efficient.py', 'rb') as fp:
    KNN = imp.load_module(
        'KNN', fp, '1.KNN_Efficient.py',
        ('.py', 'rb', imp.PY_SOURCE))    
from KNN import KNN  

# 3. Main function: 
if __name__ == '__main__':
    
    # 3.1 Generate XOR dataset:
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100) # 100 in class 0 & 100 in class 1

    # 3.2 Display the data:
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.title('XOR Data')

    # 3.3 KNN Fit & Predict:
    # K = 3
    model = KNN(3)
    model.fit(X, Y)
    print('Accuracy:', model.score(X, Y))
    print('Good accuracy!')