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


z = FrankeFunction(x, y)+np.random.normal(size=n)
z = np.ravel(z)	


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
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

X = CreateDesignMatrix_X(x,y,2)

def beta(X,z):
	return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))

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
	return 1- np.sum((z-z_tilde)**2)/np.sum((z-np.mean(z_tilde))**2)

def MSE(z, z_tilde):
	n = np.size(z_tilde)
	return np.sum((z-z_tilde)**2)/n

def RelativeError(z, z_tilde):
	return abs((z-z_tilde)/z)

def VarianceBeta(X,z):
	varZ = np.var(z)
	return  varZ * (X.T.dot(X))**(-1)

def SDBeta(X,z):
	return np.sqrt((VarianceBeta(X,z)))


def ErrorBars(X,z):
	####### error bars and plots
	betaArray = np.array(beta(X,z))

	zScore = stats.norm.ppf(0.95)
	sdArray = []
	x_value = []
	for i in range(len(betaArray)):
		Xs = X[:, i]
		sd = SDBeta(Xs, z)
		sdArray = np.append(sdArray,sd)
		x_value = np.append(x_value, i)

	#print(zScore * sdArray/ np.sqrt(X.size))
	yerr =  2*( zScore * sdArray / np.sqrt(X.size))


	#xerr = 0.1
	plt.errorbar(x_value,betaArray, yerr,lw=1)
	plt.show()

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

"""X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

beta = beta(X_train, z_train) # Matrix inversion
#clf = skl.LinearRegression().fit(X_train, z_train)
#z_tilde = clf.predict(X_train)

z_tilde = X_train.dot(beta) # Matrix inversion

print("Training R2: ")
print(R2(z_train, z_tilde))
print("Training MSE: ")
print(MSE(z_train, z_tilde))

z_predict = X_test.dot(beta) # Matrix inversion
#z_predict = clf.predict(X_test)
print("Test R2 :")
print(R2(z_test, z_predict))
print("Test MSE: ")
print(MSE(z_test, z_predict))

#plt.scatter(z_test, z_predict) #ideally this should be a straight line 
#plt.show() """

########## Cross validation k-space ##############
splits = 5
#X = np.arange(10, 20)

def Ridge_hm(X, z, lamb):
	I = np.identity(X.shape[1])
	beta_ridgeOLS = np.linalg.inv(X.T.dot(X) +lamb*I).dot(X.T.dot(z)) 
	
	return beta_ridgeOLS


def Kfold_hm(X,z, lamb):
	#Model tells us if we are using OLS or SVD-Ridge or Lasso
	X_k = np.split(X, splits)
	z_k = np.split(z, splits)


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

		if (lamb == 0):
			beta_train = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T.dot(z_train))
		
		else:
			beta_train = Ridge_hm(X_train, z_train,lamb)

		#beta_test = np.linalg.inv(X_test.T.dot(X_test)).dot(X_test.T.dot(z_test))
		z_tilde = X_train.dot(beta_train)
		z_predict = X_test.dot(beta_train)

		MSE_train_i = MSE(z_tilde, z_train)
		MSE_test_i = MSE(z_predict, z_test)

		MSE_train = np.append(MSE_train, MSE_train_i)
		MSE_test = np.append(MSE_test, MSE_test_i)
	return MSE_test, MSE_train

#MSE_test, MSE_train = Kfold_hm(X,z)

#print("MSE test: ", MSE_test)
#print("MSE train: ", MSE_train)

def MSE_Mean_Kfold(X,z, lamb):

	#Home made Kfold-MSE-mean
	MSE_test, MSE_train = Kfold_hm(X, z, lamb)
	MSE_train_mean = np.mean(MSE_train,axis=0)
	MSE_test_mean = np.mean(MSE_test,axis=0)

	#Scikit K-fold-MSE-mean
	kfold = KFold(n_splits=splits,shuffle=True)
	clf = skl.LinearRegression().fit(X, z)
	estimated_mse_sklearn = cross_val_score(clf,X,z, scoring="neg_mean_squared_error",cv=kfold)
	estimated_mse_sklearn = np.mean(-estimated_mse_sklearn)

	#print("MSE Scikit.Learn: ", estimated_mse_sklearn)
	#print("MSE train mean: ", MSE_train_mean)
	#print("MSE test mean: ", MSE_test_mean)

	return estimated_mse_sklearn, MSE_train_mean, MSE_test_mean


def scikitLearn_Ridge(X, z, nlambdas, lambdas):

## Cross-validation with scikitlearn and Ridge with k folds
	poly = PolynomialFeatures(degree = 2)
	k = 5
	kfold = KFold(n_splits = k)
	

	estimated_mse_sklearn = np.zeros(nlambdas)
	i = 0
	for lmb in lambdas:
		ridge = skl.Ridge(alpha = lmb)
		#X = poly.fit_transform(x[:, np.newaxis])
		estimated_mse_folds = cross_val_score(ridge, X, z,scoring='neg_mean_squared_error', cv=kfold)
		estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

		i += 1

	return estimated_mse_sklearn

def Plot_nthPoly_MSE_Mean(z,p): # p is max polynominal
	MSE_train_mean = []
	MSE_test_mean = []
	MSE_sci = []

	complex = np.arange(0,p+1)
	for i in range(p+1):

		X = CreateDesignMatrix_X(x, y, i) # Using the mesh x,y defined on the top
		MSE_train_mean_i = MSE_Mean_Kfold(X,z,0)[1]
		MSE_test_mean_i = MSE_Mean_Kfold(X,z,0)[2]
		MSE_sci_i = MSE_Mean_Kfold(X,z,0)[0]

		MSE_train_mean = np.append(MSE_train_mean, MSE_train_mean_i)
		MSE_test_mean = np.append(MSE_test_mean, MSE_test_mean_i)
		MSE_sci = np.append(MSE_sci, MSE_sci_i)

	#print(MSE_train_mean)
	#print(MSE_test_mean)
	#print(MSE_sci)

	plt.plot(complex, MSE_train_mean)
	plt.plot(complex, MSE_test_mean)
	plt.plot(complex, MSE_sci)

	plt.legend(['MSE train mean','MSE test mean', 'MSE sci'], loc='upper left')

	plt.show()

# This function plots the mean of k-folds MSE with respect to different lambdas, with given poly degree	
def Plot_nthLambda_MSE_Mean(z,p,nlambdas):
	MSE_train_mean = []
	MSE_test_mean = []
	#MSE_sci = []
	X = CreateDesignMatrix_X(x,y,p)
	lambdas = np.logspace(-3, 5, nlambdas)
	for l in lambdas:
		#ridge = Ridge(alpha=l)
		MSE_train_mean_i = MSE_Mean_Kfold(X,z,l)[1]
		MSE_test_mean_i = MSE_Mean_Kfold(X,z,l)[2]

		MSE_train_mean = np.append(MSE_train_mean, MSE_train_mean_i)
		MSE_test_mean = np.append(MSE_test_mean, MSE_test_mean_i)

	estimated_mse_sklearn = scikitLearn_Ridge(X,z, nlambdas, lambdas)	
	plt.plot(np.log10(lambdas), MSE_train_mean)
	plt.plot(np.log10(lambdas), MSE_test_mean)
	plt.plot(np.log10(lambdas), estimated_mse_sklearn, label= "cross_val_score")
	#plt.plot(complex, MSE_sci)

	plt.legend(['MSE train mean','MSE test mean', 'MSE Scikit'], loc='upper left')
	plt.xlabel("log10(lambda)")
	plt.ylabel("mse")
	plt.show()

Plot_nthLambda_MSE_Mean(z,2, 500)



#estimated_mse_sklearn = scikitLearn_Ridge(X,z)
#print(scikitLearn_Ridge(X, z))
#plt.figure()
#plt.plot(np.log10(lambdas), estimated_mse_sklearn, label= "cross_val_score")


#plt.legend()
#plt.show()

"""
def ConfidenceInterval(X,betanr, z, conf):  # n is n*n = 20*20 = 400 in this case
	b = beta(X, z)[betanr]
	X = X[:, betanr]
	sd = SDBeta(X,z)
	n = len(X)
	zScore = stats.norm.ppf(conf)
	#print("zscore: ", zScore)
	start = b - zScore * sd / np.sqrt(n)
	end = b + zScore * sd / np.sqrt(n)
	return start, end



def printFunctionStats(betanr,precentageOfNormalPDF):
	p = precentageOfNormalPDF
	print("Variance of the estimator: ", VarianceBeta(X[:, betanr], z))
	print("SD of the estimator: ", SDBeta(X[:, betanr], z))
	s, e = ConfidenceInterval(X, betanr, z, p)
	print("Confidence interval length", e-s)
	return
"""

#printFunctionStats(1, 0.999)
#print(clf.coef_, clf.intercept_)
"""
#VarianceBeta(z,X[0])

#print(len((X[:, 0])))
 #Plot the surface.
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#linewidth=0, antialiased=False)
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#linewidth=0, antialiased=False)
#Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
#x = np.random.rand(100)
#y = 5*x*x+0.1*np.random.randn(100)





# poly1d constructs a polynomial can use then p.r to find the roots, p.c=coeff
# p.order displaying the order and so on... 
#p = np.poly1d(np.polyfit(x.T, y,2))


"""