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
from sklearn.neural_network import MLPRegressor


from NeuralNetwork import NeuralNetwork as NN

np.set_printoptions(threshold=sys.maxsize)

seed = 3000
np.random.seed(seed)

x = np.arange(0, 1, 0.02)
y = np.arange(0, 1, 0.02)


n = x.size
x, y = np.meshgrid(x,y)

def threeSquares(low_n, high_n, l_perc, h_perc):
    for n in range(low_n, high_n+1):
        a = 1
        while a * a <= n:
            b = 1
            while b**2 <= n:
                c = 1
                while c**2 < n:
                    step = str(1/m.sqrt(n))
                    integer, floating = step.split(".")
                    if (a*a + b*b + c*c == n and (l_perc*n <= a**2 + b**2 <= h_perc*n) and 0.5 <= a**2/b**2 <= 1.5 and len(floating) <= 12) :
                        print("floating", floating)
                        print("")
                        print(a, "^2 + ", b, "^2 + ",c,"^2 + ", " n = " ,n)
                        print(a**2/n, " + ", b**2/n, " + ", c**2/n, " = 1")
                        print("step size", 1/m.sqrt(n))
                    c+=1
                b+=1
            a+=1


#threeSquares(1000,5000, 0.1, 0.6)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x,y)
z_noise = FrankeFunction(x,y)

z = np.ravel(z)
z_noise = np.ravel(z_noise)
noise = np.random.randn(z_noise.shape[0])*0.00001

z_noise += noise

shape = (x.shape[0]**2,1)
z.shape = shape
z_noise.shape = shape

def shuffle(X, z):
    shuffle_ind = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffle_ind)
    Xshuffled = np.zeros(X.shape)
    zshuffled = np.zeros(z.shape)
    for ind in range(X.shape[0]):
        Xshuffled[ind] = X[shuffle_ind[ind]]
        zshuffled[ind] = z[shuffle_ind[ind]]
    return Xshuffled, zshuffled

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
    #print("X: ", np.argmax(X))
    #print("z: ", z[np.argmax(X)])
    #X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.36, random_state=seed, shuffle=False)
    #X_test, X_val, z_test, z_val = train_test_split(X_test, z_test, test_size=0.36, random_state=seed, shuffle=False) 
    #X_train_shuffle, X_test_shuffle, z_train_shuffle, z_test_shuffle = train_test_split(X, z, test_size=0.36, random_state=seed, shuffle=True)
    #X_test_shuffle, X_val_shuffle, z_test_shuffle, z_val_shuffle = train_test_split(X_test_shuffle, z_test_shuffle, test_size=0.36, random_state=seed, shuffle=True) 
    #print("z_train: ", z_train)
    #X_train_shuffle, z_train_shuffle = shuffle(X_train, z_train)
    #X_test_shuffle, z_test_shuffle = shuffle(X_test, z_test)
    #X_val_shuffle, z_val_shuffle = shuffle(X_val, z_val)
    #print("z_train_shuffle: ",z_train_shuffle)  
    #print("X_train: ", np.argmax(X_train_shuffle))
    #print("z_train: ", z_train[np.argmax(X_train_shuffle)])     
    
    return X #X_train, X_test, X_train_shuffle, X_test_shuffle, X_val_shuffle, z_train, z_test, z_train_shuffle, z_test_shuffle, z_val_shuffle

X = CreateDesignMatrix_X(x,y,4)


#X, X_train, X_test, X_train_shuffle, X_test_shuffle, X_val_shuffle, z_train, z_test, z_train_shuffle, z_test_shuffle, z_val_shuffle = CreateDesignMatrix_X(x,y,4)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.36, random_state=seed, shuffle=False)





def R2( z, z_pred):
    return 1- (np.sum((z-z_pred)**2)/np.sum((z-np.mean(z))**2))

# MSE 
def MSE(z, z_tilde):
    n = np.size(z_tilde)
    return np.sum((z-z_tilde)**2)/n


#iterations, cols, etamax, etamin, lmbdmax, lmbdmin, batch_sizemax, batch_sizemin, hiddenmax,hiddenmin
#iterations must be larger than cols. Cols is how many of each of the parameters there will be in the given intervals.
# print("parameters: ",hypertuning(1000, 1000, 1, -6, 1, -12, 100, 1, 500, 1))

# eta, lmb, batchsize, hidden_neurons, epochs
#[4.33148322e-03 3.75469422e-11 1.00000000e+00 4.22000000e+02 1.97000000e+02]

# epochs = 183
# batch_size = 140
# eta = 1.09749877
# n_hidden_neurons = 11
# lmbd = 4.03701726e-04

#1.68743568e-01 8.39312950e-12 9.10000000e+01 3.90000000e+02
# 2.04000000e+02

def Franke_plot(X,X_train,X_test, z_train, eta=0, lmbd=0,batch_size =0, n_hidden_neurons = 0, epochs = 0):

    eta = 4.33148322e-03
    lmbd = 3.75469422e-11
    batch_size = 2
    n_hidden_neurons = 422
    epochs = 197
    # For these parameters above we got: MSE = 0.004922969949345497 R2-score =0.9397964833194705


    n_categories = 1

    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train_and_validate()
    z_pred = dnn.predict_probabilities(X)
    print('X: ', X)
    print(mean_squared_error(z, z_pred))
    print(r2_score(z, z_pred))
    xsize = x.shape[0]
    ysize = y.shape[0]

    rows = np.arange(ysize)
    cols = np.arange(xsize)

    [X, Y] = np.meshgrid(cols, rows)


    z_mesh = np.reshape(z, (ysize, xsize))
    z_predict_mesh = np.reshape(z_pred, (ysize, xsize))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    

    ax = fig.axes[0]
    ax.plot_surface(X, Y, z_predict_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted Franke')

    ax = fig.axes[1]
    ax.plot_surface(X, Y, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Real Franke')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
#Franke_plot(X,X_train,X_test)
    
def Franke_plot_overfitting(X_train,X_test, eta=0, lmbd=0,batch_size =0, n_hidden_neurons = 0, epochs = 0):

    eta = 4.33148322e-03
    lmbd = 3.75469422e-11
    batch_size = 2
    n_hidden_neurons = 422
    epochs = 1000
    


    n_categories = 1

    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train_and_validate(X_train, z_train, MSE_store=True)
    MSE_train = dnn.MSE_epoch()
    np.random.seed(seed)
    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train_and_validate(X_test, z_test, MSE_store=True)
    MSE_test = dnn.MSE_epoch()
    epo = np.arange(len(MSE_train))
    plt.plot(epo, MSE_test, label='MSE test')
    plt.plot(epo, MSE_train, label='MSE train')
    plt.xlabel("Number of epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs epochs")
    plt.legend()
    plt.show()

#Franke_plot_overfitting(X_train,X_test)   
#print(z_test)
print(z_train)
def Franke_plot_overfit_3D(X, X_train,X_test, z_train, z_test, eta=0, lmbd=0,batch_size =0, n_hidden_neurons = 0, epochs = 0):
    
    eta = 4.33148322e-03
    lmbd = 3.75469422e-11
    batch_size = 2
    n_hidden_neurons = 422
    epochs = 200
    n_categories = 1

    np.random.seed(seed)
    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ReLu')
    dnn.train_and_validate()
    z_pred = dnn.predict_probabilities(X_test)
    print(z_pred)
    #print("X_train: ", X_train)
    
    print(mean_squared_error(z_test, z_pred))
    print(r2_score(z_test, z_pred))
    test_size = int(np.sqrt(X_test.shape[0]))
    train_size = int(np.sqrt(X_train.shape[0]))

    rows_test = np.linspace(0,1,test_size)
    cols_test = np.linspace(0,1,test_size)

    rows_train = np.linspace(0,1,train_size)
    cols_train = np.linspace(0,1,train_size)


    [X1, Y1] = np.meshgrid(cols_test, rows_test)
    [X2, Y2] = np.meshgrid(cols_train, rows_train)

    z_predict_mesh = np.reshape(z_pred, (test_size, test_size))
    z_mesh = np.reshape(z_train, (train_size, train_size))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    # plt.figure()

    ax = fig.axes[0]
    ax.plot_surface(X1, Y1, z_predict_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted test')

    ax = fig.axes[1]
    ax.plot_surface(X2, Y2, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted train')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
    

Franke_plot_overfit_3D(X,X_train,X_test, z_train, z_test)


def Franke_plot_scikit(X, z, X_train, z_train):
    eta = 4.33148322e-03
    lmbd = 3.75469422e-11
    batch_size = 2
    n_neurons_layer1 = 221
    n_neurons_layer2 = 221
    epochs = 197
    # For these parameters we got: MSE = 0.004557564498595149 and R2-score= 0.9442650649634291
    n_categories = 1

    dnn = MLPRegressor(hidden_layer_sizes=(422),activation='relu', solver='sgd', alpha=lmbd, batch_size=batch_size, 
        learning_rate_init=eta, 
        max_iter=epochs,random_state=seed)
    
    dnn.fit(X_train, z_train)
    z_pred = dnn.predict(X)

    print(mean_squared_error(z, z_pred))
    print(r2_score(z, z_pred))
    xsize = x.shape[0]
    ysize = y.shape[0]

    rows = np.arange(ysize)
    cols = np.arange(xsize)

    [X, Y] = np.meshgrid(cols, rows)


    z_mesh = np.reshape(z, (ysize, xsize))
    z_predict_mesh = np.reshape(z_pred, (ysize, xsize))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    ax = fig.axes[0]
    ax.plot_surface(X, Y, z_predict_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted Franke')

    ax = fig.axes[1]
    ax.plot_surface(X, Y, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Real Franke')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()

#Franke_plot_scikit(X,z, X_train,z_train)

#epochs = 1000
#batch_size = 10
#eta = 0.001
#n_hidden_neurons = 100 #[1, 10, 20, 25, 40, 50]
#lmbd = 0.0
n_categories = 1 

def hypertuning(iterations, cols, etamax, etamin, lmbdmax, lmbdmin, batch_sizemax, batch_sizemin, hiddenmax,hiddenmin, polymin, polymax):

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
    hyper[4] = np.random.randint(polymin, polymax, size=cols, dtype='int')
    hyper[5] = np.zeros((cols))
    

    
    for i in range(rows-1):
        np.random.shuffle(hyper[i])

    n_categories = 1
    

    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]
        X, X_train, X_test, X_train_shuffle, X_test_shuffle, X_val_shuffle, z_train, z_test, z_train_shuffle, z_test_shuffle, z_val_shuffle = CreateDesignMatrix_X(x,y, int(hyper[4][it]))
        epochs = 1000

        
        dnn = NN(X_train_shuffle, z_train_shuffle, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='MSE', activation='sigmoid', activation_out='ELU')
        dnn.train_and_validate(X_val_shuffle,z_val_shuffle, MSE_store = False)
        #MSE = dnn.MSE_epoch()
        #epo = np.arange(len(MSE))
        #plt.plot(epo, MSE)
        #plt.show()
        z_pred = dnn.predict_probabilities(X_test)
        #print(z_pred)

        MSE_array[it] = mean_squared_error(z_test,z_pred)
        hyper[5][it] = dnn.epoch +1
        if (it%m.ceil((iterations/40))==0):
            print('Iteration: ', it)
            t = round((time.time() - start_time))
            if t >= 60:
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
                print("Estimated minutes left: ", int((t/it)*(iterations-it)/60))
            else:
                print("--- %s sec ---" %int(t))
    MSE_best_index = np.argmin(MSE_array)
    MSE_best = np.min(MSE_array)
    print("MSE array: ", MSE_array)
    print("best index: ",MSE_best_index)
    print("best MSE: ", MSE_best)
    final_hyper = hyper[:,MSE_best_index]

    print("parameters: eta, lmbd, batch, hidden, poly, epochs ", final_hyper)
    eta_best = final_hyper[0]
    lmbd_best = final_hyper[1]
    batch_size_best = final_hyper[2]
    n_hidden_neurons_best = final_hyper[3]
    poly_best = final_hyper[4]
    epochs_best = final_hyper[5]
    Franke_plot(X, X_train, X_test, z_train,eta=eta_best, lmbd=lmbd_best,
        batch_size =batch_size_best, n_hidden_neurons = n_hidden_neurons_best, epochs = epochs_best)
    Franke_plot_overfit_3D(X, X_train,X_test, z_train, z_test, eta=eta_best, 
        lmbd=lmbd_best,batch_size =batch_size_best, n_hidden_neurons = n_hidden_neurons_best, epochs = epochs_best)
    return hyper[:,MSE_best_index]

#hypertuning(2, 2, 1, -6, -4, -14, 100, 1, 100, 1, 2, 3)




