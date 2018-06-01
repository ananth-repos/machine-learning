# Implementation to demonstrate overfitting:
# In this code same dataset is fitted for varying degree (or flexibility)
# and test/train score is plotted vs flexibility to understand overfitting.
# First a plot is created to compare the predictions for various degrees.
# Next a plot is created to compare MSEs for different deg.

# Code Organization:
    # 1. Import all relevant libraries.
    # 2. Define make_poly function that creates polynomial features based on deg.
    # 3. Fit function to calculate weights.
    # 4. Fit & Display function that makes predictions/plots.
    # 5. Get MSE Function to calculate mean-squared errors.
    # 6. Plotting MSE from test & training dataset.
    # 7. Main function with data generation & loop for varying flexibility.
    
# 1. Imports:
import numpy as np
import matplotlib.pyplot as plt

# 2. Make poly function that creates polynomial features based on deg:
def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d+1))
    return np.vstack(data).T

# 3. Fit fucntion to calcualte weights:
def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# 4. Fit and display functions that predicts & plots the outputs:
def fit_and_display(X, Y, sample, deg):
    N = len(X)
    # Choose random sample
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]
    
    # Plot training data
    plt.figure(1)
    plt.scatter(Xtrain, Ytrain,label = 'Training data for deg %d' % deg)
    plt.legend()
    

    # Fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # display the polynomial
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
   
    plt.figure(2)
    plt.plot(X, Y)
    
    plt.figure(2)
    plt.plot(X, Y_hat,label = deg)
    plt.scatter(Xtrain, Ytrain)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Predictions vs Actual')

# 5. Calculate mean sq-error:
def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)

# 6. Plot training vs test data function:
def plot_train_vs_test_curves(X, Y, sample = 20, max_deg=20):
    N = len(X)
    # Choose random sample for train test split:
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    test_idx = [idx for idx in range(N) if idx not in train_idx]
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []
    # Loop through each deg to calculate MSEs:
    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.figure(3)
    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.title('MSE vs. Deg - Overfitting')
    # Now how the test MSE drops after certain deg which indicates overfitting
    
    # Plot just training mse to see the trend
    plt.figure(4)
    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.xlabel('Degree')
    plt.ylabel('MSE')
    plt.title('MSE vs. Deg - Overfitting')
    # Now how its gets constant after a certain deg
    

if __name__ == "__main__":
    # Generate random data & plot:
    N = 100
    X = np.linspace(0, 6*np.pi, N)
    Y = np.sin(X)

    plt.figure(1)
    plt.plot(X, Y,label = 'actual data')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Random data used in this code')

    for deg in (5, 6, 7, 8):
        fit_and_display(X, Y, 10, deg)
    plot_train_vs_test_curves(X, Y)
