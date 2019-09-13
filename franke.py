import numpy as np 
import sklearn.linear_model as skl
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.linalg as scl
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split


#fig = plt.figure()
#ax = fig.gca(projection='3d')
# Make data.
#n = 40000
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


"""
def beta2(X,z):
	a = X.T.dot(X)
	#print("a :",a)
	b = X.T.dot(z)
	#print("b :",b)
	return (a*b)**(-1)
"""
def OLS_inv(X, z):
	beta1 = beta(X,z)
	z_tilde = X.dot(beta1)
	return z_tilde

#singular value decomposition 
def ols_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y

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
#plt.errorbar(x_value,betaArray, yerr,lw=1)
#plt.show()

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

X_k = np.split(X, splits)
z_k = np.split(z, splits)

#print(z_k)
MSE_tilde = []
MSE_predict = []
for i in range(splits):

	X_train = X_k
	z_train = z_k
	

	
	X_test = X_k[i]
	X_train = np.delete(X_train, i , 0)
	X_train = np.concatenate(X_train)
	z_test = z_k[i]
	z_train = np.delete(z_train, i , 0)
	z_train = np.ravel(z_train)
	#print(X_train)
	#print(z_train.size)
	
	beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(z))
	z_tilde = X_train.dot(beta)
	z_predict = X_test.dot(beta)
	
	MSE_tilde_i = MSE(z_train, z_tilde)
	MSE_predict_i = MSE(z_test, z_predict)
	MSE_tilde = np.append(MSE_tilde, MSE_tilde_i)
	MSE_predict = np.append(MSE_predict, MSE_predict_i)

print(MSE_tilde)	
print(MSE_predict)

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