import math as m
from random import random, seed
import numpy as np 
from sklearn.model_selection import train_test_split


""" In this script there are functions that are used with both data sets and some that are only used in Franke
"""
seed = 3000



train_percentage = 0.64
test_percentage =  0.2304
validation_percentage = 0.1296

split_train_test = 1 - train_percentage
split_test_val = validation_percentage/split_train_test


# Producing the Franke function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# A code to find the perfect squares if one need to
def perfectSquares(low_n, high_n, l_perc, h_perc, squares):

    for n in range(low_n, high_n+1):
        a = 1
        while a * a <= n:
            b = 1
            while b**2 <= n:
                if squares == 2:
                    step = str(1 / m.sqrt(n))
                    integer, floating = step.split(".")
                    if (l_perc*n <= a**2 <= h_perc*n and (a**2 + b**2 == n) and m.sqrt(n).is_integer() and len(floating) <= 15):
                        print(a, "^2 + ", b, "^2 "," n = ", n)
                        print(a ** 2 / n, " + ", b ** 2 / n, " = 1")
                        print("step size", 1 / m.sqrt(n))
                elif squares == 3:
                    c = 1
                    while c**2 < n:
                        step = str(1/m.sqrt(n))
                        integer, floating = step.split(".")
                        if (a*a + b*b + c*c == n and (l_perc*n <= a**2 + b**2 <= h_perc*n) and 0.5 <= a**2/b**2 <= 1.5 and len(floating) <= 12) :
                            print("")
                            print(a, "^2 + ", b, "^2 + ",c,"^2 ", " n = " ,n)
                            print(a**2/n, " + ", b**2/n, " + ", c**2/n, " = 1")
                            print("step size", 1/m.sqrt(n))
                        c+=1
                b+=1
            a+=1

#perfectSquares(400,625, 0.01, 0.6, 3)

# Making the design matrix for the Franke function: 
def CreateDesignMatrix_X(z, x, y, n ):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)      
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
       
    X, z_, indicies = shuffle(X, z)
    X_train, X_test, z_train, z_test = train_test_split(X, z_, test_size=split_train_test, random_state=seed, shuffle=False)
    X_test, X_val, z_test, z_val = train_test_split(X_test, z_test, test_size=split_test_val, random_state=seed, shuffle=False)

    return X, X_train, X_test, X_val, z_train, z_test, z_val, indicies

#Shuffles the data by indencies 
def shuffle(X, z):
    shuffle_ind = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffle_ind)
    Xshuffled = np.zeros(X.shape)
    zshuffled = np.zeros(z.shape)
    for ind in range(X.shape[0]):
        Xshuffled[ind] = X[shuffle_ind[ind]]
        zshuffled[ind] = z[shuffle_ind[ind]]
    return Xshuffled, zshuffled, shuffle_ind     

#Backshuffles the data after it has been shuffled if needed for plotting: 
def backshuffle(X, z, X_train_s, X_test_s, z_train_s ,z_test_s ,indicies):
    train_indicies, test_val_indicies = train_test_split(indicies, test_size=split_train_test, random_state=seed, shuffle=False)
    test_indicies = train_test_split(test_val_indicies, test_size=split_test_val, random_state=seed, shuffle=False)[0]

    X_train = np.zeros((X.shape))
    X_test = np.zeros((X.shape))
    
    
    for ind in range(train_indicies.size):
        X_train[train_indicies[ind]] = X_train_s[ind]
       
    for ind in range(test_indicies.size):
        X_test[test_indicies[ind]] = X_test_s[ind]
        

    return X_train, X_test           