import numpy as np 
import sklearn.linear_model as skl
import sys
#import sklearn.linear_model as Ridge
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from inspect import signature
import time
import math as m


from NeuralNetwork import NeuralNetwork as NN

np.set_printoptions(threshold=sys.maxsize)

seed = 3000
np.random.seed(seed)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)


n = x.size
x, y = np.meshgrid(x,y)



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)
#print(z.shape)
z = np.ravel(z)

noise = np.random.randn(z.shape[0])#*0.001
#z += noise
shape = (x.shape[0]**2,1)
z.shape = shape


### Code taken from Piazza from Morten Hjort-Jensen 

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

X = CreateDesignMatrix_X(x,y,4)


X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=seed)
X_test, X_val, z_test, z_val = train_test_split(X_test, z_test, test_size= 0.5, random_state=seed)



def R2( z, z_pred):
    return 1- (np.sum((z-z_pred)**2)/np.sum((z-np.mean(z))**2))

# MSE 
def MSE(z, z_tilde):
    n = np.size(z_tilde)
    return np.sum((z-z_tilde)**2)/n

epochs = 200
#batch_size = 10
#eta = 0.001
#n_hidden_neurons = 100 #[1, 10, 20, 25, 40, 50]
#lmbd = 0.0
n_categories = 1 #z_train.shape[0]

def hypertuning(iterations, cols, etamax, etamin, lmbdmax, lmbdmin, batch_sizemax, batch_sizemin, hiddenmax,hiddenmin):
    start_time = time.time()
    if (cols < iterations):
        cols = iterations
        print("cols must be larger than 'iterations. Cols is set equal to iterations")
    sig = signature(hypertuning)
    rows = int(len(sig.parameters)/2)
    hyper = np.zeros((rows, cols))
    MSE_array = np.zeros(iterations)
    hyper[0] =  np.logspace(etamin, etamax, cols)
    hyper[1] = np.logspace(lmbdmin, lmbdmax, cols)
    hyper[2] = np.round(np.linspace(batch_sizemin, batch_sizemax, cols, dtype='int'))
    hyper[3] = np.round(np.linspace(hiddenmin, hiddenmax, cols))
    hyper[4] = np.zeros((cols))
    #print(hyper)
    for i in range(rows-1):
        np.random.shuffle(hyper[i])
        #print(np.apply_along_axis(np.random.shuffle, 1, hyper[i]))

    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]


        dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                 n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='MSE', activation='sigmoid', activation_out='ELU')

        dnn.train(X_val)
        MSE_val = dnn.MSE_epoch(z_val)

        best_pred_epoch = np.argmin(MSE_val)

        dnn_test = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=best_pred_epoch+1, batch_size=batch_size,
                  n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                  cost_grad='MSE', activation='sigmoid', activation_out='ELU')
        dnn_test.train(X_test)

        # kan jo bare bruke predict probabilities på siste her

        z_pred = dnn_test.y_predict_epoch[best_pred_epoch]

        MSE_array[it] = mean_squared_error(z_test,z_pred)
        hyper[4][it] = best_pred_epoch
        print(it)
        if (it%m.ceil((iterations/40))==0):
            t = round((time.time() - start_time))
            if t >= 60:
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
            else:
                print("--- %s sec ---" %int(t))
    MSE_best_index = np.argmin(MSE_array)
    MSE_best = np.min(MSE_array)
    print("MSE array: ", MSE_array)
    print("best index: ",MSE_best_index)
    print("best MSE: ", MSE_best)

    return hyper[:,MSE_best_index]
#iterations, cols, etamax, etamin, lmbdmax, lmbdmin, batch_sizemax, batch_sizemin, hiddenmax,hiddenmin
#iterations must be larger than cols. Cols is how many of each of the parameters there will be in the given intervals.
print("parameters: ",hypertuning(1000, 1000, 1, -6, 1, -12, 100, 1, 500, 1))

# eta, lmb, batchsize, hidden_neurons, epochs
#[4.33148322e-03 3.75469422e-11 1.00000000e+00 4.22000000e+02 1.97000000e+02]

# epochs = 183
# batch_size = 140
# eta = 1.09749877
# n_hidden_neurons = 11
# lmbd = 4.03701726e-04

def Franke_plot(X,X_train,X_test):
    eta = 4.33148322e-03
    lmbd = 3.75469422e-11
    batch_size = 1
    n_hidden_neurons = 422
    epochs = 197

    n_categories = 1

    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs+1, batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train(X)
    z_pred = dnn.y_predict_epoch[epochs]

    xsize = x.shape[0]
    ysize = y.shape[0]

    rows = np.arange(ysize)
    cols = np.arange(xsize)

    [X, Y] = np.meshgrid(cols, rows)


    z_mesh = np.reshape(z, (ysize, xsize))
    z_predict_mesh = np.reshape(z_pred, (ysize, xsize))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    # plt.figure()

    ax = fig.axes[0]
    ax.plot_surface(X, Y, z_predict_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted terrain cut')

    ax = fig.axes[1]
    ax.plot_surface(X, Y, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Terrain cut')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()

    #z_pred_train = dnn.predict_probabilities(X_train)
    #z_pred_val = dnn.predict_probabilities(X_val)

# Franke_plot(X,X_train,X_test)

#MSE_train = MSE(z_train, z_pred_train)
"""eta_vals = np.arange(1,4, 0.2)#np.logspace(-1, 1, 20)#[1e-8, 1e-6,1e-4,1e-2,1e-2,1e-1]
#print(eta_vals)
train_accuracy = np.zeros((len(n_hidden_neurons), len(eta_vals)))
test_accuracy = np.zeros((len(n_hidden_neurons), len(eta_vals)))
for i, neuron in enumerate(n_hidden_neurons):
    for j, eta in enumerate(eta_vals):
    
    #print("Eta: ", eta_vals[i])
        dnn = NeuralNetwork(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=neuron, n_categories=n_categories,
                    cost_grad = 'MSE', activation = 'sigmoid') #n_categories=n_categories


        dnn.train()
        z_train_pred = dnn.feed_forward_out(X_train)
        z_test_pred =dnn.feed_forward_out(X_test)
        train_accuracy[i][j] = r2_score( z_train, z_train_pred)
        test_accuracy[i][j] = r2_score( z_test, z_test_pred)



fig, ax = plt.subplots(figsize=(6,15))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_xticklabels(np.round(eta_vals,2))
ax.set_yticklabels(n_hidden_neurons)
ax.set_title("Training Accuracy")
ax.set_ylabel("hidden neurons")
ax.set_xlabel("$\eta$")
plt.show()
"""
#eta_vals = np.logspace(-6, -1, 10) #[1e-6, 1e-4, 1e-2, 1e-1,1]#np.logspace(-5, 1, 7)
#lmbd_vals = np.zeros()
#train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
#test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

"""for i in range(len(eta_vals)):
 #   for j in range(len(lmbd_vals)):
        #z_predict = 0
    dnn = NeuralNetwork(X_train_scaled, z_train, eta=i, lmbd=j, epochs=epochs, batch_size=batch_size,
          n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
          cost_grad = 'MSE', activation = 'sigmoid')
        z_predict = dnn.predict_probabilities(X_test_scaled)#[:,1:2]
     
        #train_accuracy[i][j] = R2(z_train, )
        test_accuracy[i][j] = r2_score( z_test_scaled, z_predict)"""
"""
#print(z_test)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()    

def nthLambda_error( error, nlambdas):
    lambdas = np.logspace(-8, 5, nlambdas)


    error_train = np.zeros((nlambdas))
    error_test = np.zeros((nlambdas)) 
    error_test_l = 0
    for lamb in range(len(lambdas)):
        dnn = NeuralNetwork(X_train, z_train, eta=eta, lmbd=lamb, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons,
                    cost_grad = 'MSE', activation = 'sigmoid')  
        dnn.train()
        z_predict = dnn.predict_probabilities(X_test)[:,1:2]
        if (error == 'MSE'):
            #error_train_l = MSE(z_train, z_predict)
            error_test_l = MSE(z_test, z_predict)

        if(error == "R2"):
            #error_train_l = R2(z_train, z_predict)
            error_test_l = R2(z_test, z_predict)
                
        #error_train[lamb] = error_train_l
        error_test[lamb] = error_test_l

    #plt.plot(np.log10(lambdas), error_train_mean)
    plt.plot(np.log10(lambdas), error_test)
    plt.show()"""


