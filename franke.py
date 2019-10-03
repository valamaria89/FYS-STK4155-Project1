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

colors = ['#1f77b4', '#1f77b4', '#aec7e8','#aec7e8','#ff7f0e','#ff7f0e','#d62728','#d62728','#2ca02c','#2ca02c','#98df8a','#98df8a','#ffbb78','#ffbb78']


seed = 4000

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

n = x.size
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    #print(term1.size)
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4




z = FrankeFunction(x, y)
z = np.ravel(z)
np.random.seed(seed)
#noiseadj = 0.001
noise = np.random.randn(z.shape[0])
#z += noise*noiseadj

#print(z)


# def this: 
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

X = CreateDesignMatrix_X(x,y,2)


def beta(X,z):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(z))

def OLS_inv(X, z):
    beta1 = beta(X,z)
    z_tilde = X.dot(beta1)
    return z_tilde


#singular value decomposition 
def OLS_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #Gives the betas for the SVD
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y



#print("z_OLS_inv: ", z_OLS_inv)
#print("beta_svd: ", beta_SVD)

#check against scikitLearn : 
def check_scikitLearn(X, z):
    clf = skl.LinearRegression().fit(X, z)
    z_tilde = clf.predict(X)
    return z_tilde

#z_tilde = OLS_inv(X,z)


#Error analysis
def R2(z, z_tilde):
    #print('z ',z)
    #print('zean', np.mean(z))
    #print('final ',z-np.mean(z))
    #if((sum(z-np.mean(z)))==0): print(z)
    return 1- np.sum((z-z_tilde)**2)/np.sum((z-np.mean(z))**2)

def MSE(z, z_tilde):
    n = np.size(z_tilde)
    return np.sum((z-z_tilde)**2)/n

def RelativeError(z, z_tilde):
    return abs((z-z_tilde)/z)

def VarianceBeta(X,z):
    varZ = 1*noiseadj #np.var(z)
    return  np.diag(varZ * np.linalg.pinv(X.T.dot(X)))


def SDBeta(X,z):
    return np.sqrt((VarianceBeta(X,z)))


def Ridge_hm(X, z, lamb):
    I = np.identity(X.shape[1])
    beta_ridge = np.linalg.pinv(X.T.dot(X) +lamb*I).dot(X.T.dot(z))

    return beta_ridge   

def Lasso_SciKit_Beta(X, z, lamb):
    model_lasso = skl.Lasso(alpha=lamb, fit_intercept=False, normalize=True, tol=0.1)
    model_lasso.fit(X, z)
    beta_Lasso =  model_lasso.coef_    
    return beta_Lasso 

def OLS_Scikit_Beta(X, z):
    reg = LinearRegression(fit_intercept=False, normalize=True).fit(X,z)
    beta = reg.coef_
    return beta

def ErrorBars(X, z, lamb):
    ####### error bars and plots
    betaArray = np.array(Lasso_SciKit_Beta(X,z, lamb))

    zScore = stats.norm.ppf(0.95)
    sdArray = []
    x_value = []
#for i in range(len(betaArray)):
    #Xs = X[:, i]
    sdArray = SDBeta(X, z)
    
    x_value = np.arange(len(betaArray))
    yerr =  ( zScore * sdArray) # np.sqrt(X.shape[0]))
#yerr = sdArray
    return betaArray, x_value


def plott_errorbar(z):
    lambdas = [0.001, 0.01, 0.1, 0.2, 0.3]
    nrow = 2
    ncol = 3
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# β', ha='center')
    fig.text(0.01, 0.5, 'β values', va='center', rotation='vertical')
    for i, ax in enumerate(fig.axes):
        #ax.set_ylabel('beta values')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' p = %s'%(i+1))
        X = CreateDesignMatrix_X(x,y,(i+1))
        for l in range(len(lambdas)):
            betaArray, x_value = ErrorBars(X,z, lambdas[l])
            ax.plot(x_value,betaArray,lw=1,linestyle=':',marker='o', label= '$\lambda$ = %s'%lambdas[l])

    ax.legend(loc='upper center', bbox_to_anchor=(0.3, -0.09), fancybox=True, shadow=True, ncol=5)
   
    plt.show()    

#plott_errorbar(z)

############ Error analysis ##################
"""print("Variance score R2 code: ", R2(z,z_tilde))
print("Mean Squared Error code: ", MSE(z, z_tilde))
#print("Relative Error: ", RelativeError(z, z_tilde))

#ScikitLearn Error Analysis
print("Mean squared error: %.2f" % mean_squared_error(z, z_tilde))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(z,z_tilde))
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(z, z_tilde))"""

######### Splitting data into train & test ################


def Train_Test_Reg(X,z, model, lamb):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    if (model == 'OLS'):
        beta_train = beta(X_train, z_train) # Matrix inversion
        
    elif (model=='Ridge'):
        beta_train = Ridge_hm(X_train, z_train,lamb)

    elif (model=='Lasso'):
        beta_train = Lasso_SciKit_Beta(X_train, z_train, lamb)    

    z_tilde = X_train.dot(beta_train)
    z_predict = X_test.dot(beta_train) # Matrix inversion
    """print("Training R2: ")
    print(R2(z_train, z_tilde))
    print("Training MSE: ")
    print(MSE(z_train, z_tilde))

    
    #z_predict = clf.predict(X_test)
    print("Test R2 :")
    print(R2(z_test, z_predict))
    print("Test MSE: ")
    print(MSE(z_test, z_predict))"""
    
    return X_train, X_test, z_train, z_test, z_predict, z_tilde



def Plot_Train_Test_OLS(z, model='OLS'):

    noiseadj = [0.5, 0.1, 0.01]
    nrow = 1
    ncol = 3
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, 'z_test', ha='center')
    fig.text(0.01, 0.5, 'z_predict', va='center', rotation='vertical')
    for i, ax in enumerate(fig.axes):

        z_new = z+noise*noiseadj[i]
        z_test, z_predict = Train_Test_OLS(X,z_new)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.scatter(z_test, z_predict)
    
   
    plt.show()

#Plot_Train_Test_OLS(z)      

########## Cross validation k-space ##############
splits = 5
#X = np.arange(10, 20)

def Kfold_hm(X,z, model, lamb):
    #Model tells us if we are using OLS or SVD-Ridge or Lasso

    #shuffling the data
    shuffle_ind = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffle_ind)

    Xshuffled = np.zeros(X.shape)
    zshuffled = np.zeros(X.shape[0])
    for ind in range(X.shape[0]):

        Xshuffled[ind] = X[shuffle_ind[ind]]
        zshuffled[ind] = z[shuffle_ind[ind]]


    X_k = np.split(Xshuffled, splits)
    z_k = np.split(zshuffled, splits)


    MSE_train = []
    MSE_test = []
    for i in range(splits):

        X_train = X_k
        X_train = np.delete(X_train, i, 0)
        X_train = np.concatenate(X_train)

        z_train = z_k
        z_train = np.delete(z_train, i, 0)
        z_train = np.ravel(z_train)

        X_test = X_k[i]
        z_test = z_k[i]

        if (model== 'OLS'):
            beta_train = beta(X_train, z_train) #OLS_Scikit_Beta(X_train, z_train)
            #beta(X_train, z_train)
        
            #np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))

        elif (model=='Ridge'):
            beta_train = Ridge_hm(X_train, z_train,lamb)

        elif (model=='Lasso'):
            beta_train = Lasso_SciKit_Beta(X_train, z_train, lamb)
             

        #beta_test = np.linalg.pinv(X_test.T.dot(X_test)).dot(X_test.T.dot(z_test))
        z_tilde = X_train.dot(beta_train) #+ model_lasso.intercept_
        z_predict = X_test.dot(beta_train) #+ model_lasso.intercept_

        MSE_train_i = R2(z_tilde, z_train)
        MSE_test_i = R2(z_predict, z_test)

        MSE_train = np.append(MSE_train, MSE_train_i)
        MSE_test = np.append(MSE_test, MSE_test_i)

    return MSE_test, MSE_train, z_test, z_predict


#print("MSE test: ", MSE_test)
#print("MSE train: ", MSE_train)

def MSE_ScikitLearn(X,z):
    kfold = KFold(n_splits=splits,shuffle=False)
    clf = skl.LinearRegression().fit(X, z)
    estimated_mse_sklearn = cross_val_score(clf,X,z, scoring="neg_mean_squared_error",cv=kfold)
    estimated_mse_sklearn = -estimated_mse_sklearn
    return estimated_mse_sklearn


def MSE_Mean_Kfold(X,z, model, lamb):

    #Home made Kfold-MSE-mean
    MSE_test = Kfold_hm(X, z, model, lamb )[0]
    MSE_train = Kfold_hm(X, z, model , lamb)[1]
    MSE_train_mean = np.mean(MSE_train,axis=0)
    MSE_test_mean = np.mean(MSE_test,axis=0)

    #Scikit K-fold-MSE-mean
    kfold = KFold(n_splits=splits,shuffle=True, random_state=seed)
    clf = skl.LinearRegression(fit_intercept=False, normalize=True).fit(X, z)
    estimated_mse_sklearn = cross_val_score(clf,X,z, scoring="neg_mean_squared_error",cv=kfold)
    estimated_mse_sklearn = np.mean(-estimated_mse_sklearn)

    #print("MSE Scikit.Learn: ", estimated_mse_sklearn)
    #print("MSE train mean: ", MSE_train_mean)
    #print("MSE test mean: ", MSE_test_mean)

    return estimated_mse_sklearn, MSE_train_mean, MSE_test_mean



def scikitLearn_Lasso(X, z, nlambdas, lambdas):
   # k = 5
   # kfold = KFold(n_splits = k)
    #lambdas = np.logspace(-3, 5, nlambdas)

    estimated_mse_sklearn = np.zeros(nlambdas)
    i = 0
    
    for lmb in lambdas:
        
        k = 5
        kfold = KFold(n_splits = k, shuffle=True,random_state=seed)
        model_lasso = skl.Lasso(alpha=lmb, fit_intercept=False, normalize=True)
        #lasso = linear_model.Lasso()
        #model_lasso.fit(X,z)
        #X = model_lasso.predict(X)
        estimated_mse_folds = cross_val_score(model_lasso, X, z,scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)
        i += 1

    return estimated_mse_sklearn

#print(scikitLearn_Lasso(X,z, 1))


def scikitLearn_Ridge(X, z, nlambdas, lambdas):

## Cross-validation with scikitlearn and Ridge with k folds
    

   # X = X - np.mean(X,axis=0)
    #poly = PolynomialFeatures(degree = 0)
    estimated_mse_sklearn = np.zeros(nlambdas)
    i = 0
    for lmb in lambdas:

        k = 5
        kfold = KFold(n_splits = k, shuffle=True, random_state=seed)
        ridge = skl.Ridge(alpha = lmb, fit_intercept=False, normalize=True)
        #X = poly.fit_transform(x[:, np.newaxis])
        #print(X.size)
        estimated_mse_folds = cross_val_score(ridge, X, z,scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

        i += 1


    return estimated_mse_sklearn


"""nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)
estimated_mse_sklearn = scikitLearn_Ridge(x,z, nlambdas, lambdas)

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()"""
def Plot_nthPoly_regular( z, error, model, p, nlambdas, pol_LasRid):
    
    
    

    noiseadj = [ 1, 0.5, 0.1, 0.01]
    nrow = 1
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# Polynomials', ha='center')
    fig.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
    
    
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        if (model=='OLS'):

            complex = np.arange(0,p+1)
            error_train_mean = np.zeros((p+1))
            error_test_mean = np.zeros((p+1))
            for pol in range(p+1):
                X = CreateDesignMatrix_X(x, y, (pol+1)) # Using the mesh x,y defined on the top
                X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z, model, nlambdas)

                if (error=='MSE'):
                    error_train_pol = MSE(z_tilde, z_train)
                    error_test_pol = MSE(z_predict, z_test)

                elif (error=='R2'):
                    error_train_pol = R2(z_tilde, z_train)
                    error_test_pol = R2(z_predict, z_test)

                
                error_train_mean_pol =  np.mean(error_train_pol)
                error_test_mean_pol =  np.mean(error_test_pol)
                

                error_train_mean[pol] =  error_train_mean_pol
                error_test_mean[pol] = error_test_mean_pol
            
        elif(model=='Ridge' and 'Lasso'):

            lambdas = np.logspace(-8, 5, nlambdas)
            error_train_mean = np.zeros((nlambdas))
            error_test_mean = np.zeros((nlambdas))   
            X = CreateDesignMatrix_X(x, y, pol_LasRid)

            for l in range(len(lambdas)):
                
                 # Using the mesh x,y defined on the top
                if (model=='Ridge'):
                    X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z, model, lambdas[l])

                elif (model=='Lasso'):
                    X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z, model, lambdas[l])
          
                if (error=='MSE'):
                    error_train_l = MSE(z_tilde, z_train)
                    error_test_l = MSE(z_predict, z_test)

                elif (error=='R2'):
                    error_train_l = R2(z_tilde, z_train)
                    error_test_l = R2(z_predict, z_test)

                error_train_mean_l = np.mean(error_train_l)
                error_test_mean_l = np.mean(error_test_l)

                error_train_mean[l] = error_train_mean_l
                error_test_mean[l] = error_test_mean_l
            
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(complex, error_train_mean)
        ax.plot(complex, error_test_mean)
        #ax.plot(complex, MSE_sci)
        ax.legend(['MSE train mean','MSE test mean'], loc='upper left')
    
    

    plt.show()
Plot_nthPoly_regular(z, 'MSE', 'OLS', 10, 0, 0)    

def Plot_nthLambda_nthPoly_Regular(z, error, p = 3, lamb = 0, model = 'Ridge'):
    steps = 5
    maxlamb = np.log10(lamb)
    lambdas = np.logspace(-6, maxlamb, steps)
    lambdas = [round(l,5) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
    #lambdas = [0.003,0.03,0.3,3,30]
    complexity = np.arange(0,p+1)

   # if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
    #elif model == 'Ridge':
    error_train_array = np.zeros((steps,p + 1))
    error_test_array = np.zeros((steps,p + 1))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    for i in range(len(lambdas)):
        for p in range(p+1):
            X = CreateDesignMatrix_X(x, y, p)
            if (model=='Ridge'):
                    X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z, model, lambdas[i])

            elif (model=='Lasso'):
                X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z, model, lambdas[i])
      
            if (error=='MSE'):
                error_train_i = MSE(z_tilde, z_train)
                error_test_i = MSE(z_predict, z_test)

            elif (error=='R2'):
                error_train_i = R2(z_tilde, z_train)
                error_test_i = R2(z_predict, z_test)

            error_train_array[i][p] =  np.mean(error_train_i)
            error_test_array[i][p] =  np.mean(error_test_i)
    

        plt.plot(complexity, error_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
        plt.plot(complexity, error_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.xlabel('Model Complexity for Ridge without resampling')
    plt.ylabel('Prediction Error (MSE)')
    plt.show()

#Plot_nthLambda_nthPoly_Regular(z, 'MSE', 5, 1, 'Ridge')

def Plot_nthPoly_MSE_Mean(z,p): # p is max polynominal
    MSE_train_mean = np.zeros((p+1))
    MSE_test_mean = np.zeros((p+1))
    MSE_sci = np.zeros((p+1))
    noiseadj = [ 0.5, 0.25, 0.1, 0.01]
    nrow = 1
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# Polynomials', ha='center')
    fig.text(0.01, 0.5, 'R^{2} score', va='center', rotation='vertical')
    
    complex = np.arange(0,p+1)
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        for pol in range(p+1):
            X = CreateDesignMatrix_X(x, y, pol) # Using the mesh x,y defined on the top
            #print("polynomial: ", i, " Determinant: ", np.linalg.det(X.T.dot(X)))
            MSE_train_mean_pol = MSE_Mean_Kfold(X,z_new,'OLS', 0)[1]
            MSE_test_mean_pol = MSE_Mean_Kfold(X,z_new,'OLS', 0)[2]
            MSE_sci_pol = MSE_Mean_Kfold(X,z_new,'OLS', 0)[0]

            MSE_train_mean[pol] =  MSE_train_mean_pol
            MSE_test_mean[pol] = MSE_test_mean_pol
            #MSE_sci[pol] = MSE_sci_pol
            
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(complex, MSE_train_mean)
        ax.plot(complex, MSE_test_mean)
        #ax.plot(complex, MSE_sci)
        ax.legend(['R2 train mean','R2 test mean'], loc='upper left')
    
    

    plt.show()

 
# This function plots the mean of k-folds MSE with respect to different lambdas, with given poly degree 
def Plot_nthLambda_MSE_Mean(z,p, model, nlambdas):
    
    lambdas = np.logspace(-8, 5, nlambdas)


    X = CreateDesignMatrix_X(x,y,p)
    #print(lambdas)
    MSE_train_mean = np.zeros((nlambdas))
    MSE_test_mean = np.zeros((nlambdas))
    #print(MSE_train_mean.size)
    noiseadj = [1, 0.5, 0.1, 0.01]
    nrow = 1
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, 'log10(lambda)', ha='center')
    fig.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
    
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        for l in range(len(lambdas)):
           # print(l)
            MSE_train_mean_l = MSE_Mean_Kfold(X,z_new, model, lambdas[l])[1]
            MSE_test_mean_l = MSE_Mean_Kfold(X,z_new,model, lambdas[l])[2]

            MSE_train_mean[l] = MSE_train_mean_l
            MSE_test_mean[l] = MSE_test_mean_l

        #estimated_mse_sklearn_ridge = scikitLearn_Ridge(X, z, nlambdas, lambdas)
        #estimated_mse_sklearn_lasso = scikitLearn_Lasso(X, z, nlambdas, lambdas)
        #print(MSE_test_mean)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(np.log10(lambdas), MSE_train_mean)
        ax.plot(np.log10(lambdas), MSE_test_mean)
    #ax.plot(complex, MSE_sci)
        ax.legend(['MSE Lasso train','MSE Lasso test'], loc='upper left')
        #print(np.amin(MSE_test_mean))
    #plt.plot(np.log10(lambdas), MSE_train_mean)
    #plt.plot(np.log10(lambdas), MSE_test_mean)
    #plt.plot(np.log10(lambdas), estimated_mse_sklearn_ridge)
    #plt.plot(np.log10(lambdas), estimated_mse_sklearn_lasso)
    #plt.plot(complex, MSE_sci)

    #plt.legend(['MSE train mean_ridge','MSE test mean_ridge', 'MSE SK Ridge', "MSE SK Lasso"], loc='upper left')
    #plt.xlabel("log10(lambda)")
    #plt.ylabel("mse")
    plt.show()


#Plot_nthLambda_MSE_Mean(z,4, 'Ridge', 500)

def train_test(X, z, lamb):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    beta_train = np.linalg.pinv(X_train.T.dot(X_train) + lamb * np.identity(X_train.shape[1])).dot(X_train.T.dot(z_train))
    z_predict = X_test.dot(beta_train)
    return z_test, z_predict

def Plot_VarBias_nthpoly(z,p):
    error = np.zeros(p+1)
    bias = np.zeros(p+1)
    variance = np.zeros(p+1)
    Error_biva = np.zeros(p+1)
    complexity = np.arange(0,p+1)
    noiseadj = [2, 1, 0.5, 0.1, 0.01]
    nrow = 1
    ncol = 5
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# Polynomials', ha='center')
    fig.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
    
    complex = np.arange(0,p+1)
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        for pol in range(p+1):
            X = CreateDesignMatrix_X(x, y, pol)
            z_test,z_predict = train_test(X, z_new, 0)
            error[pol] =  np.mean((z_test - z_predict)**2)
            bias[pol] = np.mean((z_test - np.mean(z_predict))**2)
            variance[pol] = np.mean( np.var(z_predict))
            Error_biva[pol] = bias[pol] + variance[pol] + np.mean(noise)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(complex, Error_biva, label='Error_biva')
        ax.plot(complex, error, label='Error')
        ax.plot(complex, bias, label='bias')
        ax.plot(complex, variance, label='Variance')
        ax.legend()

    plt.show()

#Plot_VarBias_nthpoly(z,10)
def Plot_nthLambda_nthPoly(z, p = 3, lamb = 0, model = 'Ridge'):
    steps = 5
    maxlamb = np.log10(lamb)
    lambdas = np.logspace(-6, maxlamb, steps)
    lambdas = [round(l,5) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
    #lambdas = [0.003,0.03,0.3,3,30]
    complexity = np.arange(0,p+1)

   # if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
    #elif model == 'Ridge':
    MSE_train_array = np.zeros((steps,p + 1))
    MSE_test_array = np.zeros((steps,p + 1))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    for i in range(len(lambdas)):
        for p in range(p+1):
            X = CreateDesignMatrix_X(x, y, p)
            MSE_train_array[i][p], MSE_test_array[i][p] = MSE_Mean_Kfold(X, z, model, lambdas[i])[1:3]

    

        plt.plot(complexity, MSE_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
        plt.plot(complexity, MSE_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.xlabel('Model Complexity for Lasso')
    plt.ylabel('Prediction Error (MSE)')
    plt.show()
    #elif model == 'Lasso': # Only MSE_test. Cross_val_score(sklearn) only returns MSE_test.
        
      #  MSE_train_array = np.zeros((steps,p + 1))
     #   MSE_test_array = np.zeros((steps, p + 1))
       # for i in range(len(lambdas)):
        #    for p in range(p + 1):
         #       k = 5
          #      X = CreateDesignMatrix_X(x, y, p)
           #     MSE_train_array[i][p], MSE_test_array[i][p] = MSE_Mean_Kfold(X, z, model, lambdas[i])[1:3]

            #plt.plot(complexity, MSE_test_array[i], label='$\lambda$ = %s' % lambdas[i])
       # plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.77), fancybox=True, shadow=True)
       # plt.subplots_adjust(right=0.75)
       # plt.xlabel('Model Complexity')
       # plt.ylabel('Prediction Error (MSE)')
       # plt.show()

# print("Error: ", error[i])
# print('Bias^2:', bias[i])
# print('Var:', variance[i])
# print('{} >= {} + {} = {}'.format(error[i], bias[i], variance[i], bias[i]+variance[i]))




########## All plots #############
#ErrorBars(z, 2)
#Train_Test_OLS(X,z)
#MSE_test, MSE_train = Kfold_hm(X,z, 0)
#MSE_ScikitLearn_OLS = MSE_ScikitLearn(X,z)
#print("MSE test: ", MSE_test)
#print("MSE train: ", MSE_train)
#print("MSE Scikit: ", MSE_ScikitLearn_OLS)
#Plot_VarBias_nthpoly(z,10)
#Plot_nthLambda_MSE_Mean(z,2, 500)
#Plot_nthPoly_MSE_Mean(z,10)
#Plot_nthLambda_nthPoly(z,5, 1, 'Lasso')

