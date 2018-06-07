# Using KNN for the alternating dots problem.
# In this implementation K is set to 3 delibrately to make the KNN fail.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Import KNN function from standard implementation.
    # 3. Main function.
        # 3.1 Generate alternating dots dataset.
        # 3.2 Display the data.
        # 3.3 KNN Fit & Predict (K = 3).
        
# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt
import imp

# 2. Import KNN function from standard implementation:
with open('0.KNN.py', 'rb') as fp:
    KNN = imp.load_module(
        'KNN', fp, '0.KNN.py',
        ('.py', 'rb', imp.PY_SOURCE))    
from KNN import KNN  

# 3. Main function: 
if __name__ == '__main__':
    
    # 3.1 Generate Alternating Dots dataset:
    # Grid size (8,8)
    width = 8
    height = 8
    N = width * height
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0
    start_t = 0
    # Alternate dots:
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t + 1) % 2 # alternate between 0 and 1
        start_t = (start_t + 1) % 2

    # 3.2 Display the data:
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.title('Alternating Dots')
    
    # 3.3 KNN Fit & Predict:
    model = KNN(3)
    model.fit(X, Y)
    print('Accuracy:', model.score(X, Y))
    print('KNN not useful for alternating dots!')
    print('Set K = 1 & try again!')