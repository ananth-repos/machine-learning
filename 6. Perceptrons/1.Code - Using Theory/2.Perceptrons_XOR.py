# Using Perceptron on XOR data.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Import Perceptron function.
    # 3. Main function.
        # 3.1 Generate XOR data.
        # 3.2 Display.
        # 3.3 Perceptron function call.
        
# 1. Import all relevant libraries:
import imp
import numpy as np
import matplotlib.pyplot as plt

# 2. Import Perceptron function from vectorized implementation:
with open('0.Perceptrons_Linear.py', 'rb') as fp:
    Perceptron = imp.load_module(
        'Perceptron', fp, '0.Perceptrons_Linear.py',
        ('.py', 'rb', imp.PY_SOURCE))    
from Perceptron import Perceptron 

# 3. Main Function: 
if __name__ == '__main__':
        
    # 3.1 Generate XOR data:
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100) # 100 in class 0 & 100 in class 1
    Y[Y == 0] = -1
    
    # 3.2 Display the data:
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.title('XOR Data')
    
    # 3.3 Perceptron Function Call:
    model = Perceptron()
    model.fit(X, Y)
    print("XOR accuracy:", model.score(X, Y))
    print('Performace not so great!')

