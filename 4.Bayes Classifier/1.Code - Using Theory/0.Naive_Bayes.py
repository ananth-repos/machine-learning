# Implementation of Naive Bayes(NB) classifier.
# Tested on MNIST dataset.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. NB class definition.
        # 2.1 Fit function.
        # 2.2 Predict function.
        # 2.3 Score function.
    # 3. Main Function.
        # 3.1 Read the MNIST dataset.
        # 3.2 Test train split.
        # 3.3 NB function call.
        
# 1. Import all relevant libraries:
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import multivariate_normal as mvn
from pathlib import Path

# 2. Naive Bayes class definition:
class NaiveBayes(object):
    # Variables: gaussians, priors
    # Methods: fit, predict, score
    
    # 2.1 Fit function:
    def fit(self, X, Y, smoothing=1e-2): # smoothing to avoid div by 0 for variance.
        self.gaussians = dict() # tuples (mean, variance)
        self.priors = dict()
        labels = set(Y) 
        for c in labels: # loop through each labels.
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    # 2.2 Predict function:
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in (self.gaussians).items():
            mean, var = g['mean'], g['var']
            # Use scipy logpdf function because exp slows it down
            # Covariance can be specified as vector
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)
    
    # 2.3 Score function:
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

# 3. Main Function: 
if __name__ == '__main__':
       
    # 3.1 Read the MNIST dataset:
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
    
    # 3.2 Test train split:
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # 3.3 NB function call:
    model = NaiveBayes() # Initilalize the model
    
    # Training:
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    # Prediction on training dataset:
    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    # Prediction on test dataset:
    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
