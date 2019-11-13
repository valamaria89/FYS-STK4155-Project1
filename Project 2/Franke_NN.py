import numpy as np 
import sklearn.linear_model as skl
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from random import random, seed
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.linalg as scl
from scipy import stats
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
import math as m
from sklearn.neural_network import MLPRegressor
from hypertuning import hypertuning_franke
from Misc import FrankeFunction, shuffle, backshuffle, perfectSquares, CreateDesignMatrix_X
from NeuralNetwork import NeuralNetwork as NN

np.set_printoptions(threshold=sys.maxsize)

seed = 3500
np.random.seed(seed)

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
n = x.size
x, y = np.meshgrid(x,y)

z = FrankeFunction(x,y)
z = np.ravel(z)

noise = np.random.randn(z.shape[0])*0.1
z += noise

shape = (x.shape[0]**2,1)
z.shape = shape

X, X_train, X_test, X_val, z_train, z_test, z_val, indicies  = CreateDesignMatrix_X(z,x, y ,4)

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
    print(mean_squared_error(z, z_pred))
    print(r2_score(z, z_pred))
    xsize = x.shape[0]
    ysize = y.shape[0]

    rows = np.arange(ysize)
    cols = np.arange(xsize)

    [X1, Y1] = np.meshgrid(cols, rows)


    z_mesh = np.reshape(z, (ysize, xsize))
    z_predict_mesh = np.reshape(z_pred, (ysize, xsize))

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    

    ax = fig.axes[0]
    ax.plot_surface(X1, Y1, z_predict_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted Franke')

    ax = fig.axes[1]
    ax.plot_surface(X1, Y1, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Real Franke')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()

#Franke_plot(X,X_train,X_test,z_train)
    
def Franke_plot_overfitting(X_train,X_test, eta=0, lmbd=0,batch_size =0, n_hidden_neurons = 0, epochs = 0):

    eta = 3.16227766e-01
    lmbd = 2.68269580e-08
    batch_size = 1
    n_hidden_neurons = 57
    epochs = 1000
    n_categories = 1

    np.random.seed(seed)
    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train_and_validate(X_val, z_val, MSE_store=True, validate =False)
    MSE_val, MSE_train = dnn.MSE_epoch()
    epo = np.arange(len(MSE_train))
    plt.plot(epo, MSE_val, label='MSE val')
    plt.plot(epo, MSE_train, label='MSE train')
    plt.xlabel("Number of epochs")
    plt.ylabel("MSE")
    plt.title("MSE vs epochs")
    plt.legend()
    plt.show()

#Franke_plot_overfitting(X_train,X_test)   

def Franke_plot_fit_3D(X, z, X_train,X_test, z_train, z_test, indicies, eta=0, lmbd=0,batch_size =0, n_hidden_neurons = 0, epochs = 0):

    eta = 3.16227766e-01
    lmbd = 2.68269580e-08
    batch_size = 1
    n_hidden_neurons = 57
    epochs = 91
    n_categories = 1

    # With the parameters above we got these values for MSE and R2 for 10 000 points and no noise:
    #MSE z_test_predict:  0.0012524282064846637
    #R2 z_test_predict:  0.9851769055209932
    #MSE z_train_predict  0.0012329368999059254
    #R2 z_train_predict 0.9848613850303888

    # With parameters above we got these values for MSE and R2 for 10 000 poinst with noise 0.1:
    #MSE z_test_predict:  0.027152301040701644
    #R2 z_test_predict:  0.71125231579683
    #MSE z_train_predict  0.027160342432850662
    #R2 z_train_predict 0.7113015602592969


    np.random.seed(seed)
    dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=int(epochs), batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'MSE', activation = 'sigmoid', activation_out='ELU')
    dnn.train_and_validate()

    z_pred_test_unscaled = dnn.predict_probabilities(X_test)
    z_pred_train_unscaled = dnn.predict_probabilities(X_train)
    print("MSE z_test_predict: ", mean_squared_error(z_test, z_pred_test_unscaled))
    print("R2 z_test_predict: ", r2_score(z_test, z_pred_test_unscaled))
    print("MSE z_train_predict ", mean_squared_error(z_train, z_pred_train_unscaled))
    print("R2 z_train_predict", r2_score(z_train, z_pred_train_unscaled))

    X_train_, X_test_ = backshuffle(X, z, X_train,X_test,z_train,z_test,indicies)

    z_pred_test = dnn.predict_probabilities(X_test_)

    ysize_test = int(np.sqrt(z_pred_test.shape[0]))
    xsize_test = int(np.sqrt(z_pred_test.shape[0]))
    rows_test = np.linspace(0, 1, ysize_test)
    cols_test = np.linspace(0, 1, xsize_test)
    z_predict_mesh_test = np.reshape(z_pred_test, (ysize_test, xsize_test))
    [X1, Y1] = np.meshgrid(cols_test, rows_test)

    ####################

    z_pred_train = dnn.predict_probabilities(X_train_)

    ysize_train = int(np.sqrt(z_pred_train.shape[0]))
    xsize_train = int(np.sqrt(z_pred_train.shape[0]))
    rows_train = np.linspace(0, 1, ysize_train)
    cols_train = np.linspace(0, 1, xsize_train)
    z_predict_mesh_train = np.reshape(z_pred_train, (ysize_train, xsize_train))
    [X2, Y2] = np.meshgrid(cols_train, rows_train)

    ####################

    ysize = int(np.sqrt(z.shape[0]))
    xsize = int(np.sqrt(z.shape[0]))
    rows = np.linspace(0, 1, ysize)
    cols = np.linspace(0, 1, xsize)
    z_mesh= np.reshape(z, (ysize, xsize))
    [X3, Y3] = np.meshgrid(cols, rows)
    
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'})

    ax = fig.axes[0]
    ax.plot_surface(X1, Y1, z_predict_mesh_test, cmap=cm.viridis,rstride=1, cstride=1, antialiased = True, linewidth=0)
    ax.set_title('Fitted test')

    ax = fig.axes[1]
    ax.plot_surface(X2, Y2, z_predict_mesh_train, cmap=cm.viridis, linewidth=0)
    ax.set_title('Fitted train')

    ax = fig.axes[2]
    ax.plot_surface(X3, Y3, z_mesh, cmap=cm.viridis, linewidth=0)
    ax.set_title('Real Franke')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()



#Franke_plot_overfit_3D(X, z, X_train,X_test, z_train, z_test, indicies)


def Franke_plot_scikit(X, z, X_train, z_train):
    eta = 3.16227766e-01
    lmbd = 2.68269580e-08
    batch_size = 1
    n_hidden_neurons = 57
    epochs = 91
    n_categories = 1

    dnn = MLPRegressor(hidden_layer_sizes=(422),activation='logistic', solver='sgd', alpha=lmbd, batch_size=batch_size, 
        learning_rate_init=eta, 
        max_iter=epochs,random_state=seed)
    
    dnn.fit(X_train, z_train)
    z_pred = dnn.predict(X_test)

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

#def hypertuning_franke(z, iterations, cols, etamin, etamax, lmbdmin, lmbdmax, batch_sizemin, batch_sizemax, hiddenmin,hiddenmax, polymin, polymax)
#hypertuning_franke(z, x, y, 20, 20, -6, 1, -14, -4, 1, 100, 1, 80, 3, 11, plot_MSE = True)

