# Using Perceptron on MNIST data.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Import Perceptron function.
    # 3. Main function.
        # 3.1 Get MNIST data.
        # 3.2 Perceptron function call.
        
# 1. Import all relevant libraries:
import numpy as np
import pandas as pd
import imp
from datetime import datetime
from pathlib import Path

# 2. Import Perceptron function from vectorized implementation:
with open('0.Perceptrons_Linear.py', 'rb') as fp:
    Perceptron = imp.load_module(
        'Perceptron', fp, '0.Perceptrons_Linear.py',
        ('.py', 'rb', imp.PY_SOURCE))    
from Perceptron import Perceptron 

# 3. Main Function: 
if __name__ == '__main__':
    
    # 3.1 Get the MNIST Data:
    p = Path(__file__).parents[2]
    df = pd.read_csv(p /'0.Data' / 'train.csv')
    np.random.seed(101) # Set seed before shuffling
    data = df.as_matrix() # DF to matrix
    np.random.shuffle(data) # Shuflle data
    X = data[:, 1:] / 255.0 # Scale input data such that it is between [0,1] from [0,255]
    Y = data[:, 0]
    limit = 2000
    # Limit the number of data points to speed things up:
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
        
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1
    
    # 3.2 Perceptron Function Call:
    model = Perceptron()
    
    # Fit:
    t0 = datetime.now()
    model.fit(X, Y, learning_rate=1e-2)
    print("MNIST train accuracy:", model.score(X, Y))


