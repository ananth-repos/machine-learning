# Perceptron implementation on linearly separable data.

# Code Flow:
    # 1. Import all relevant libraries.
    # 2. Perceptron class definition.
        # 2.1 Fit function.
        # 2.2 Predict function.
        # 2.3 Score function.
    # 3. Main Function.
        # 3.1 Generate data.
        # 3.2 Test train split.
        # 3.3 Perceptron function call.
        
# 1. Import all relevant libraries:
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 2. Perceptron Class Definition:
class Perceptron:
    
    # 4.1 Fit function:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):       
        # initialize random weights
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        costs = [] # keep track of costs
        for epoch in range(epochs):
            # Determine misclassified samples:
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break # Done

            # Choose a random incorrect sample
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            # Calculate Cost:
            # Incorrect Rate
            c = len(incorrect) / float(N)
            costs.append(c)
        
        # Print results:
        print("final w:", self.w, "final b:", self.b, "epochs:", (epoch+1), "/", epochs)
        
        # Plot costs:
        plt.figure(2)
        plt.plot(costs)
        plt.xlabel('Epoch')
        plt.ylabel('Costs')
        plt.title('Costs vs iteration')
        
    # 4.2 Predict function:
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    # 4.3 Score function:
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

# 3. Main Function: 
if __name__ == '__main__':
    
    # 3.1 Generate Data:
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    Y = np.sign(X.dot(w) + b)
    
    # 3.2 Display the data:
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.title('Linearly Separable Data')
    
    # 3.3 Train Test Split:
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    # 3.4 Perceptron Function Call:
    model = Perceptron()
    
    # Fit:
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    # Prediction on training data:
    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    # Prediction on test data:
    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
 

