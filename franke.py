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

colors = ['#1f77b4', '#1f77b4', '#aec7e8','#aec7e8','#ff7f0e','#ff7f0e','#d62728','#d62728','#2ca02c','#2ca02c','#98df8a','#98df8a','#ffbb78','#ffbb78']


seed = 4000

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
z = np.ravel(z)
np.random.seed(seed)
noise = np.random.randn(z.shape[0])



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

X = CreateDesignMatrix_X(x,y,2)


############################ Beta-parameters ###########################

# Creates beta parameters for OLS
def beta(X,z):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(z))

# Creates beta parameters for Ridge
def Ridge_hm(X, z, lamb):
    I = np.identity(X.shape[1])
    beta_ridge = np.linalg.pinv(X.T.dot(X) +lamb*I).dot(X.T.dot(z))

    return beta_ridge 


# Creates beta parameters for Lasso
def Lasso_SciKit_Beta(X, z, lamb):
    model_lasso = skl.Lasso(alpha=lamb, fit_intercept=False, normalize=True, tol=0.1)
    model_lasso.fit(X, z)
    beta_Lasso =  model_lasso.coef_    
    return beta_Lasso     


def OLS_inv(X, z):
    beta1 = beta(X,z)
    z_tilde = X.dot(beta1)
    return z_tilde


#singular value decomposition 
def OLS_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    #Gives the betas for the SVD
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y


#check against scikitLearn : 
def check_scikitLearn(X, z):
    clf = skl.LinearRegression().fit(X, z)
    z_tilde = clf.predict(X)
    return z_tilde


########################### Error analysis ############################

# R² score 
def R2(z_tilde, z):
    return 1- (np.sum((z-z_tilde)**2)/np.sum((z-np.mean(z))**2))

# MSE 
def MSE(z, z_tilde):
    n = np.size(z_tilde)
    return np.sum((z-z_tilde)**2)/n

#Relative Error 
def RelativeError(z, z_tilde):
    return abs((z-z_tilde)/z)

#Variance for beta parameter
def VarianceBeta(X,z, noiseadj):
    varZ = 1*noiseadj 
    return  np.diag(varZ * np.linalg.pinv(X.T.dot(X)))

# Standard deviation 
def SDBeta(X,z, noiseadj):
    return np.sqrt((VarianceBeta(X,z, noiseadj)))


  

################## SciKit-Learn codes to check against home-made codes ####################

# Beta parameters for OLS
def OLS_Scikit_Beta(X, z):
    reg = LinearRegression(fit_intercept=False, normalize=True).fit(X,z)
    beta = reg.coef_
    return beta    

# MSE from Scikit-Learn with k-fold resampling method
def MSE_ScikitLearn(X,z):

    kfold = KFold(n_splits=splits,shuffle=False)
    clf = skl.LinearRegression().fit(X, z)
    estimated_mse_sklearn = cross_val_score(clf,X,z, scoring="neg_mean_squared_error",cv=kfold)
    estimated_mse_sklearn = -estimated_mse_sklearn
    return estimated_mse_sklearn

def scikitLearn_OLS(X, z,):
    #Scikit K-fold-MSE-mean
    kfold = KFold(n_splits=splits,shuffle=True, random_state=seed)
    clf = skl.LinearRegression(fit_intercept=False, normalize=True).fit(X, z)
    estimated_mse_sklearn = cross_val_score(clf,X,z, scoring="neg_mean_squared_error",cv=kfold)
    estimated_mse_sklearn = np.mean(-estimated_mse_sklearn)

# Ridge Regression from Scikit-Learn with k-fold resampling method for different lambdas 
def scikitLearn_Ridge(X, z, nlambdas, lambdas):

    estimated_mse_sklearn = np.zeros(nlambdas)
    i = 0
    for lmb in lambdas:

        k = 5
        kfold = KFold(n_splits = k, shuffle=True, random_state=seed)
        ridge = skl.Ridge(alpha = lmb, fit_intercept=False, normalize=True)
        estimated_mse_folds = cross_val_score(ridge, X, z,scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

        i += 1


    return estimated_mse_sklearn


# Lasso Regression from Scikit-Learn with k-fold resampling method for different lambdas 
def scikitLearn_Lasso(X, z, nlambdas, lambdas):
  
    estimated_mse_sklearn = np.zeros(nlambdas)
    i = 0
    
    for lmb in lambdas:
        k = 5
        kfold = KFold(n_splits = k, shuffle=True,random_state=seed)
        model_lasso = skl.Lasso(alpha=lmb, fit_intercept=False, normalize=True)
        estimated_mse_folds = cross_val_score(model_lasso, X, z,scoring='neg_mean_squared_error', cv=kfold)
        estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)
        i += 1


    return estimated_mse_sklearn


################# Functions for plotting beta parameters with conficence intervals ########################

def ErrorBars(X, z, lamb, noiseadj):
    betaArray = np.array(Beta(X, z))

    zScore = stats.norm.ppf(0.95)
    sdArray = []
    x_value = []

    sdArray = SDBeta(X, z, noiseadj)
    
    x_value = np.arange(len(betaArray))
    yerr =  ( zScore * sdArray) 

    return betaArray, x_value


def plott_errorbar(z , lambdas, noiseadj):
    nrow = 2
    ncol = 3
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# β', ha='center')
    fig.text(0.01, 0.5, 'β values', va='center', rotation='vertical')
    for i, ax in enumerate(fig.axes):
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' p = %s'%(i+1))
        X = CreateDesignMatrix_X(x,y,(i+1))
        for l in range(len(lambdas)):
            betaArray, x_value = ErrorBars(X,z, lambdas[l], noiseadj)
            ax.plot(x_value,betaArray,lw=1,linestyle=':',marker='o', label= '$\lambda$ = %s'%lambdas[l])

    ax.legend(loc='upper center', bbox_to_anchor=(0.3, -0.09), fancybox=True, shadow=True, ncol=5)
   
    plt.show()    

#lambdas= [0.001, 0.01, 0.1, 0.2, 0.3]
#plott_errorbar(z,lambdas, 1)

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


######### Splitting data into train & test without resampling ################


# Splits the data into train and test and obtains the beta parameter for all models for then to return the values needed
def Train_Test_Reg(X,z, model, lamb):
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=seed)

    if (model == 'OLS'):
        beta_train = beta(X_train, z_train) # Matrix inversion
        
    elif (model=='Ridge'):
        beta_train = Ridge_hm(X_train, z_train,lamb)

    elif (model=='Lasso'):
        beta_train = Lasso_SciKit_Beta(X_train, z_train, lamb)    

    z_tilde = X_train.dot(beta_train)
    z_predict = X_test.dot(beta_train) # Matrix inversion

    
    return X_train, X_test, z_train, z_test, z_predict, z_tilde


# Plots the relationship between z_test and z_predict 
def Plot_Train_Test_OLS(z, model='OLS', lamb=0):

    noiseadj = [0.5, 0.1, 0.01]
    nrow = 1
    ncol = 3
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, 'z_test', ha='center')
    fig.text(0.01, 0.5, 'z_predict', va='center', rotation='vertical')
    for i, ax in enumerate(fig.axes):

        z_new = z+noise*noiseadj[i]
        z_test, z_predict = Train_Test_Reg(X,z_new, model, lamb)[3:5]
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.scatter(z_test, z_predict)
    
   
    plt.show()

#Plot_Train_Test_OLS(z)      

########## Cross validation k-space ##############

splits = 5
def Kfold_hm(X,z, error, model, lamb):

    #shuffling the data
    shuffle_ind = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffle_ind)

    Xshuffled = np.zeros(X.shape)
    zshuffled = np.zeros(X.shape[0])
    for ind in range(X.shape[0]):

        Xshuffled[ind] = X[shuffle_ind[ind]]
        zshuffled[ind] = z[shuffle_ind[ind]]

    # splitting the data 
    X_k = np.split(Xshuffled, splits)
    z_k = np.split(zshuffled, splits)

    # Running the k-fold cross-validation 
    error_train = []
    error_test = []
    for i in range(splits):

        X_train = X_k
        X_train = np.delete(X_train, i, 0)
        X_train = np.concatenate(X_train)

        z_train = z_k
        z_train = np.delete(z_train, i, 0)
        z_train = np.ravel(z_train)

        X_test = X_k[i]
        z_test = z_k[i]

        #Choosing the beta parameter according to model 
        if (model== 'OLS'):
            beta_train = beta(X_train, z_train) 

        elif (model=='Ridge'):
            beta_train = Ridge_hm(X_train, z_train,lamb)

        elif (model=='Lasso'):
            beta_train = Lasso_SciKit_Beta(X_train, z_train, lamb)
        

        z_tilde = X_train.dot(beta_train) 
        z_predict = X_test.dot(beta_train) 

        if (error == 'MSE'):
        # Calculating the MSE 
            error_train_i = MSE(z_tilde, z_train)
            error_test_i = MSE(z_predict, z_test)

        elif (error=='R2'):
            error_train_i = R2(z_tilde, z_train)
            error_test_i = R2(z_predict, z_test)


        error_train = np.append(error_train, error_train_i)
        error_test = np.append(error_test, error_test_i)

    return error_test, error_train, z_test, z_predict



# Returns an array of the MSE_train_mean and MSE_test_mean from the k-fold code above 
def Error_Mean_Kfold(X,z, error, model, lamb):

    #Home made Kfold-MSE-mean
    error_test = Kfold_hm(X, z, error, model, lamb )[0]
    error_train = Kfold_hm(X, z, error, model , lamb)[1]
    error_train_mean = np.mean(error_train,axis=0)
    error_test_mean = np.mean(error_test,axis=0)

    return error_train_mean, error_test_mean



################## Plotting functions #################################

# Function calculates the MSE or R²-score for all the regression models without resampling for nth polynomial for different noises
def Plot_nthPoly_regular( z, error, model, p, nlambdas, pol_LasRid):
    
    noiseadj = [ 1, 0.5, 0.1, 0.01]
    nrow = 1
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, 'log10(lambdas)', ha='center')
    fig.text(0.01, 0.5, 'R2', va='center', rotation='vertical')
    
    
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        if (model=='OLS'):

            complex = np.arange(0,p+1)
            error_train_mean = np.zeros((p+1))
            error_test_mean = np.zeros((p+1))
            for pol in range(p+1):
                X = CreateDesignMatrix_X(x, y, (pol+1)) # Using the mesh x,y defined on the top
                X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z_new, model, nlambdas)

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
            
        elif(model=='Ridge' or 'Lasso'):

            lambdas = np.logspace(-8, 5, nlambdas)
            error_train_mean = np.zeros((nlambdas))
            error_test_mean = np.zeros((nlambdas))   
            X = CreateDesignMatrix_X(x, y, pol_LasRid)

            for l in range(len(lambdas)):
                
                 # Using the mesh x,y defined on the top
                if (model=='Ridge'):
                    X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z_new, model, lambdas[l])

                elif (model=='Lasso'):
                    X_train, X_test, z_train, z_test, z_predict, z_tilde = Train_Test_Reg(X, z_new, model, lambdas[l])
          
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
        ind = np.argmin(error_test_mean)
        print("noise: ", noiseadj[i])
        print("max lambda ", lambdas[ind])
        print("max lambda log ", np.log10(lambdas[ind]))
        print("   ")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(np.log10(lambdas), error_train_mean)
        ax.plot(np.log10(lambdas), error_test_mean)
        #ax.plot(complex, MSE_sci) np.log10(lambdas)
        ax.legend(['R2 train mean','R2 test mean'], loc='lower left')
    
    

    plt.show()
#Plot_nthPoly_regular(z, 'MSE', 'Lasso', 0, 500, 2)    


# Function plots the nth lambda for nth poly without resampling for different noises 
def Plot_nthLambda_nthPoly_Regular(z, error, noiseadj, p = 3, lamb = 0, model = 'Ridge'):
    
    z += noise*noiseadj
    steps = 7
    maxlamb = np.log10(lamb)
    lambdas = np.logspace(-8, maxlamb, steps)
    #lambdas = [round(l,5) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
    #lambdas = [1e-08, 2.15e-07, 4.64e-06,0.0001,0.0022,0.046, 1]
    lambdas = [1e-08, 1.468e-07, 2.154e-06, 3.162e-05, 0.00046, 0.00681, 0.01] # Lasso
    complexity = np.arange(1,p+1)

   
    error_train_array = np.zeros((steps,p ))
    error_test_array = np.zeros((steps,p ))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    for i in range(len(lambdas)):
        for p in range(1, p+1):
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

            error_train_array[i][p-1] =  error_train_i
            error_test_array[i][p-1] =  error_test_i
    

        plt.plot(complexity, error_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
        plt.plot(complexity, error_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.xlabel('Model Complexity for Lasso without resampling')
    plt.ylabel('MSE')
    plt.show()

#Plot_nthLambda_nthPoly_Regular(z, 'MSE', 0.1, 5, 0.1, 'Lasso')


# Plotting the MSE or the R²-score for OLS of the nth polynomial with resampling 
def Plot_nthPoly_error_Mean(z,p, error): # p is max polynominal
    error_train_mean = np.zeros((p+1))
    error_test_mean = np.zeros((p+1))
    error_sci = np.zeros((p+1))
    noiseadj = [ 1, 0.5, 0.1, 0.01]
    nrow = 1
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol)
    fig.tight_layout()
    fig.text(0.5, 0.01, '# Polynomials', ha='center')
    fig.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
    
    complex = np.arange(0,p+1)
    for i, ax in enumerate(fig.axes):
        z_new = z+noise*noiseadj[i]
        for pol in range(p+1):
            X = CreateDesignMatrix_X(x, y, (pol+1)) # Using the mesh x,y defined on the top

            #print("polynomial: ", i, " Determinant: ", np.linalg.det(X.T.dot(X)))
            error_train_mean_pol = Error_Mean_Kfold(X,z_new, error,'OLS', 0)[0]
            error_test_mean_pol = Error_Mean_Kfold(X,z_new, error, 'OLS', 0)[1]
            

            error_train_mean[pol] =  error_train_mean_pol
            error_test_mean[pol] = error_test_mean_pol
            #MSE_sci[pol] = MSE_sci_pol
            
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(complex, error_train_mean)
        ax.plot(complex, error_test_mean)
        #ax.plot(complex, MSE_sci)
        ax.legend(['MSE train mean','MSE test mean'], loc='upper left')
    
    

    plt.show()

#Plot_nthPoly_error_Mean(z,10, 'MSE') 


# This function plots the mean of k-folds MSE with respect to different lambdas, with given poly degree 
def Plot_nthLambda_error_Mean(z,p, error, model, nlambdas):
    
    lambdas = np.logspace(-8, 5, nlambdas)


    X = CreateDesignMatrix_X(x,y,p)
    
    error_train_mean = np.zeros((nlambdas))
    error_test_mean = np.zeros((nlambdas))
   
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
            error_train_mean_l = Error_Mean_Kfold(X,z_new, error, model, lambdas[l])[0]
            error_test_mean_l = Error_Mean_Kfold(X,z_new, error, model, lambdas[l])[1]

            error_train_mean[l] = error_train_mean_l
            error_test_mean[l] = error_test_mean_l

        

        ind = np.argmin(error_test_mean)
        print("noise: ", noiseadj[i])
        print("max lambda ", lambdas[ind])
        print("max lambda log ", np.log10(lambdas[ind]))
        print("   ")
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_title(' noise = %s'%noiseadj[i])
        ax.plot(np.log10(lambdas), error_train_mean)
        ax.plot(np.log10(lambdas), error_test_mean)
    
        ax.legend(['MSE Lasso train','MSE Lasso test'], loc='lower left')
      
    plt.show()


#Plot_nthLambda_error_Mean(z,2, 'MSE', 'Lasso', 500)

#This function plots different lambdas for nth polynomial with resampling k-fold 
def Plot_nthLambda_nthPoly(z, noiseadj, error, p = 3, lamb = 0, model = 'Ridge'):
    z += noise*noiseadj
    steps = 7
    maxlamb = np.log10(lamb)
    lambdas = np.logspace(-8, maxlamb, steps)
    #lambdas = [round(l,5) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
    #lambdas = [1e-05, 3.162e-05, 0.00016, 0.00052, 0.001, 0.0032,0.01]
    complexity = np.arange(1,p+1)

   
    error_train_array = np.zeros((steps,p ))
    error_test_array = np.zeros((steps,p ))
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    for i in range(len(lambdas)):
        for p in range(1,p+1):
            X = CreateDesignMatrix_X(x, y, p)
            error_train_array[i][p-1], error_test_array[i][p-1] = Error_Mean_Kfold(X, z, error, model, lambdas[i])[0:2]

    

        plt.plot(complexity, error_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
        plt.plot(complexity, error_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
    plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
    plt.subplots_adjust(right=0.75)
    plt.xlabel('Model Complexity for Lasso')
    plt.ylabel('MSE')
    plt.show()
  
#Plot_nthLambda_nthPoly(z,0.01, 'MSE', 5, 0.1, 'Lasso')  





