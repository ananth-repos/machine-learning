# Implentation of K-Nearest Neighbors classifier on MNIST dataset.
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
from sortedcontainers import SortedList
from datetime import datetime
from pathlib import Path
  
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
        # Takes in X and calcuates class Y only from training data irrespective of X.
        y = np.zeros(len(X)) 
        for i,x in enumerate(X): # test points
            sl = SortedList() # stores (distance, class) tuples
            for j,xt in enumerate(self.X): # training points
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k: # if there are less than 'k' distance values
                    sl.add( (d, self.y[j]) )
                else: 
                    # compare distance with the largest distance value in the sorted list
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j])) # Add the new distance value
            
            # Calcualte votes now:
            votes = {} 
            for _, v in sl:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0 # start with zero
            max_votes_class = -1 # Initialize class
            for v,count in votes.items(): # count number of votes for each class 0-9
                if count > max_votes: 
                    max_votes = count # update max votes
                    max_votes_class = v # udpate corresponding class
            y[i] = max_votes_class
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
    
    print('Notice how the fit time is very less compared to predict function call!')
    print('Fit just stores data. Prediction does all the work.')
    