import numpy as np 
import sklearn.linear_model as skl
#import sklearn.linear_model as Ridge
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.linalg as scl
from scipy import stats
import pandas as pd
from cycler import cycler
import matplotlib as mpl
from matplotlib.ticker import NullFormatter
from sklearn.pipeline import make_pipeline 
from sklearn.utils import resample

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    #print(term1.size)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def CreateDesignMatrix_X(x, y, n ):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)      # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X

N = 20
p = 15
N_val = int(N) # for validation set
seed = 140

np.random.seed(seed)
# create points
x = np.sort(np.random.uniform(0,1,N))
y = np.sort(np.random.uniform(0,1,N))
x, y = np.meshgrid(x,y,sparse=False)

#create validation set
x_val =  x #np.sort(np.random.uniform(0,1,N_val))
y_val = y  #np.sort(np.random.uniform(0,1,N_val))
#x_val, y_val = np.meshgrid(x_val,y_val,sparse=False)    

#create datapoints/results
z = FrankeFunction(x, y)
z_val = FrankeFunction(x_val,y_val)

# Create noise
noiseadj = 0.01

noise = noiseadj*np.random.randn(N,N)
noise_val = noiseadj*np.random.randn(N_val,N_val)

#add noise
z_noise = z+noise
z_val_noise = z_val + noise_val

#flatten for use in functions
z_n = np.matrix.ravel(z_noise) 
z_val_n = np.matrix.ravel(z_val_noise)
z_test=z_val_n[:,np.newaxis]

#polydegrees to run
complexity =np.arange(1,p+1)

n_bootstraps = 100


error= np.zeros(p)
bias =np.zeros(p)
variance =np.zeros(p)

lin = LinearRegression(fit_intercept=False)
for deg in range(1,p+1):
    
    X = CreateDesignMatrix_X(x,y,deg)
    X_val = CreateDesignMatrix_X(x_val,y_val,deg)
    z_pred = np.empty((len(z_val_n), n_bootstraps),dtype=np.float64)

    for i in range(n_bootstraps):
            x_, y_ = resample(X, z_n)
            # Evaluate the new model on the same test data each time.
            z_pred[:, i] = lin.fit(x_, y_).predict(X_val).ravel()
    error[deg-1] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[deg-1] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[deg-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    #err.append(error)
    #bi.append(bias)
    #vari.append(variance)
#max_pd = 12 #max polynomial degree to plot to
plt.figure()
plt.plot(complexity,error,'k',label='MSE')
plt.plot(complexity,bias,'b',label='Bias^2')
plt.plot(complexity,variance,'y',label='Var')
summ=np.zeros(len(variance))
for i in range(len(error)):
    summ[i]=variance[i]+bias[i]
plt.plot(complexity,summ,'ro',label='sum')

plt.xlabel('Polynomial degree')
plt.ylabel('MSE')
plt.legend()
plt.show()

