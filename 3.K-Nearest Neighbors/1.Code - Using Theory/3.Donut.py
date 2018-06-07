# Using KNN for the Donut problem.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Import KNN function from vectorized implementation.
    # 3. Main function.
        # 3.1 Generate Donut dataset.
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
    
    # 3.1 Generate Donut dataset:
    N = 200 # No. of samples
    
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
    Y = np.array([0]*(N//2) + [1]*(N//2))

    # 3.2 Display the data:
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.xlabel('Cosine angles')
    plt.ylabel('Sine angles')
    plt.title('Donut Data')

    # 3.3 KNN Fit & Predict:
    model = KNN(3)
    model.fit(X, Y)
    print('Accuracy:', model.score(X, Y))
    print('Good accuracy!')