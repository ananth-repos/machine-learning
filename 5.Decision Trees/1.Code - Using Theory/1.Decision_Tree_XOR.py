# Implementation of Simple Decision Tree for Binary output & continous vector input.
# XOR Problem.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Entropy Function.
    # 3. Tree Node class.
        # 3.1 Fit function.
        # 3.2 Find split function.
        # 3.3 Calculate Information Gain.
        # 3.4 Predict One function.
        # 3.5 Predict function.
    # 4. Decision Tree class.
    # 5. Main Function:
        # 5.1 Generate XOR dataset.
        # 5.2 Test train split.
        # 5.3 DT fucntion call.
        
# 1. Import all relevant libraries:
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle

# 2. Entropy Function:
def entropy(y):
    
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0*np.log2(p0) - p1*np.log2(p1)

# 3. Tree Node Class:
class TreeNode:
    # Methods: fit, predict, find_split, information_gain, predict_one
    
    # Initialize:
    def __init__(self, depth=0, max_depth=None):
        
        self.depth = depth
        self.max_depth = max_depth

    # 3.1 Fit Function:
    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            # Base case:
                # 1. Only 1 sample
                # 2. Node receives examples only from 1 class. Can't make split.
            
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0 # Max Information Gain
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig: # replace max_ig, best_col, best_split
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth: # leaf node
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    left_idx = (X[:,best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    # 3.2 Find split Function:
    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    # 3.3 Calculate Information Gain:
    def information_gain(self, x, y, split):
        # Assume classes are 0 and 1
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0 
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

    # 3.4 Predict One Function:
    def predict_one(self, x):
        # use "is not None" because 0 means False
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            # corresponds to having only 1 prediction
            p = self.prediction
        return p

    # 3.5 Predict Function:
    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


# 4. Decision Tree Class:
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


# 5. Main Function:
if __name__ == '__main__':
    
    # 5.1 Generate XOR dataset:
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100) # 100 in class 0 & 100 in class 1
    X, Y = shuffle(X, Y) # Shuffle data
    
    # Binary Labels:
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    # 5.2 Train Test Split:
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    # 5.3 DT Function Call:
    model = DecisionTree() 
    
    # Fit:
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    # Prediction for training dataset:
    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0))

    # Prediction for test dataset:
    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0))
