# Implementation of vectorized K-Nearest Neighbors classifier on MNIST dataset.
# This implemention is more efficient than 0.kNN.
# Also the effect of K-value on train/test scores are evaluated.
# K-value  = (1, 2, 3, 4, 5)

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. KNN class definition.
        # 2.1 Fit function - Just stores data.
        # 2.2 Predict function.
        # 2.3 Score function.
    # 3. Main Function.
        # 3.1 Read the MNIST dataset.
        # 3.2 Test train split.
        # 3.3 KNN function call - loop through each K value from 1 to 5.
        # 3.4 Plots.
        
# 1. Import all relevant libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances

# 2. KNN class definition:
class KNN(object):
    def __init__(self, k):
        self.k = k
    
    # 2.1 Fit function - Just stores data:    
    def fit(self, X, y):
        self.X = X
        self.y = y

    # 2.2 Predict function:
    def predict(self, X):
        N = len(X)
        y = np.zeros(N)

        # Returns distances in a matrix of shape (N_test, N_train)
        distances = pairwise_distances(X, self.X)
        

        # Get the minimum k elements' indexes
        # https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
        idx = distances.argsort(axis=1)[:, :self.k]

        # Get the winning votes using the index above:
        votes = self.y[idx]

        # Count the votes for each class:
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()

        return y

    # 2.3 Score Function:
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

# 3. Main Function:   
if __name__ == '__main__':
    
    # 3.1 Read the MNIST dataset:
    p = Path('.')
    df = pd.read_csv(p /'Data' / 'train.csv')
    np.random.seed(101) # Set seed before shuffling
    data = df.as_matrix() # DF to matrix
    np.random.shuffle(data) # Shuflle data
    X = data[:, 1:] / 255.0 # Scale input data such that it is between [0,1] from [0,255]
    Y = data[:, 0]
    limit = 2000
    # Limit the number of data points to speed things up:
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
        
    # 3.2 Test train split:
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    # 3.3 KNN function call:
    train_scores = [] # keep track of training scores (always 1?)
    test_scores = [] # keep track of test scores
    ks = (1,2,3,4,5) # Sweep through different Ks
    for k in ks:
        print("\nk =", k)
        knn = KNN(k) # Initilalize the model
        # Training:
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain) # Fit the model
        print("Training time:", (datetime.now() - t0)) 

        # Prediction on training dataset:
        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        # Prediction on test dataset:
        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # 3.4 Plots:
    plt.figure(1)
    plt.plot(ks, train_scores, label='train scores',marker = 'o')
    plt.plot(ks, test_scores, label='test scores',marker = 'o')
    plt.xlabel('K = 1, 2, 3, 4, 5')
    plt.ylabel('Scores')
    plt.legend()
    plt.title('Model Performance vs Flexibility')
    
    print('Notice how the quick this vector method is comprated to scalar implementation!')

