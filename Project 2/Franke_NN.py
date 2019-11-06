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
from NeuralNetwork import NeuralNetwork as NN
np.set_printoptions(threshold=sys.maxsize)

seed = 3000

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
#np.random.seed(seed)
noise = np.random.randn(z.shape[0])#*0.001
#z += noise
shape = (400,1)
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





epochs = 500
batch_size = 10
eta = 0.001
n_hidden_neurons = 100 #[1, 10, 20, 25, 40, 50]
lmbd = 0
n_categories = 1 #z_train.shape[0]

dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'MSE', activation = 'sigmoid', activation_out='sigmoid') #n_categories=n_categories

dnn1 = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'MSE', activation = 'sigmoid', activation_out='tanh')
epoc_vals = np.arange(epochs)
dnn.train(X_train)
MSE_train = dnn.MSE_epoch(z_train)

dnn1.train(X_val)
MSE_val = dnn1.MSE_epoch(z_val)
plt.plot(epoc_vals, MSE_train, label="Train")
plt.plot(epoc_vals, MSE_val, label="Val")
plt.legend()
plt.show()

#z_pred_train = dnn.predict_probabilities(X_train)
#z_pred_val = dnn.predict_probabilities(X_val)



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


