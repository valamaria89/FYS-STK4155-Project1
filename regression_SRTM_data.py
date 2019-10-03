import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
import random
from scipy import stats
import platform
from cycler import cycler
platform.architecture()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.linear_model as skl

colors = ['#1f77b4', '#1f77b4', '#aec7e8','#aec7e8','#ff7f0e','#ff7f0e','#d62728','#d62728','#2ca02c','#2ca02c','#98df8a','#98df8a','#ffbb78','#ffbb78']
# Load the terrain
terrain = imread('SRTM_Oslo_Fjorden.tif')
#terrain = imread('SRTM_Norway_1.tif')
# Show the terrain

# plt.figure()
# plt.title('Terrain over Oslo Fjorden')
# plt.imshow(terrain, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
#plt.show()


seed = 206
maxValue = np.amax(terrain)
random.seed(seed)

### Global variables
y_size = 20
x_size = 20
x = np.linspace(0, 1, x_size)
y = np.linspace(0, 1, y_size)
x, y = np.meshgrid(x, y)

# Pick a random cut of the map of certain size
col = random.randint(0,terrain.shape[1]-x_size)
row = random.randint(0,terrain.shape[0]-y_size)
cut = terrain[row:row+y_size, col:col+x_size]
z = cut
z = z.reshape(-1,1)



def CreateDesignMatrix(x,y,p):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(p)
    X = poly.fit_transform(np.concatenate((x, y), axis=1))

    return X

def beta(X,z,lamb = 0):
    z = np.ravel(z)
    return np.linalg.pinv(X.T.dot(X) + lamb * np.identity(X.shape[1])).dot(X.T.dot(z))

def Confidence(X, z, betas = None, lamb = 0, result = 'Errorbar', confidence = 0.95):

    if betas == None:
        z = z.reshape(-1, 1)
        betas = beta(X,z, lamb)
    variance_z = 1
    Variance_Beta = np.diag(variance_z * np.linalg.pinv(X.T.dot(X)))
    SD_array = np.sqrt(Variance_Beta)
    zScore = stats.norm.ppf(confidence)
    beta_n = np.arange(len(betas))

    if result == 'Errorbar':
        yerr = zScore * SD_array #/ np.sqrt(X.shape[0])     # xerr = 0.1
        plt.errorbar(beta_n, betas, yerr, lw=1,linestyle=':',marker='o')
        plt.show()
    elif result == 'SD':
        return SD_array

def ErrorAnalysis(z1,z2, error = 'MSE'):  # e.g z1 = z_train, z2 = ztilde or z1 = z_test, z2 = z_predict
    z1 = np.ravel(z1)
    z2 = np.ravel(z2)
    if error == 'MSE':
        n = np.size(z2)
        return np.sum((z1 - z2) ** 2) / n
    elif error == 'R2':
        return 1 - np.sum((z1 - z2) ** 2) / np.sum((z1 - np.mean(z1)) ** 2)
    return


def Kfold(X, z, lamb = 0, error ='MSE'):
    splits = 5
    z = np.ravel(z)

    shuffle_ind = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffle_ind)

    X_shuffled = np.zeros(X.shape)
    z_shuffled = np.zeros(X.shape[0])

    for ind in range(X.shape[0]):
        X_shuffled[ind] = X[shuffle_ind[ind]]
        z_shuffled[ind] = z[shuffle_ind[ind]]

    X_k = np.split(X_shuffled, splits)
    z_k = np.split(z_shuffled, splits)


    MSE_train = np.zeros((splits))
    MSE_test = np.zeros((splits))
    for i in range(splits):

        X_train = X_k
        X_train = np.delete(X_train, i, 0)
        X_train = np.concatenate(X_train)

        z_train = z_k
        z_train = np.delete(z_train, i, 0)
        z_train = np.ravel(z_train)

        X_test = X_k[i]
        z_test = z_k[i]
        beta_train = beta(X_train, z_train,lamb)

        z_tilde = X_train.dot(beta_train)
        z_predict = X_test.dot(beta_train)

        MSE_train[i] = ErrorAnalysis(z_train, z_tilde, error='MSE')
        MSE_test[i] = ErrorAnalysis(z_test, z_predict, error='MSE')

    MSE_train_mean = np.mean(MSE_train,axis=0)
    MSE_test_mean = np.mean(MSE_test,axis=0)

    return MSE_train_mean, MSE_test_mean

### Regression ###

##### Resampling ##### No resampling, resampling = 'none' #####  Bootstrap, resampling = 'bootstrap' ##### k-fold, resampling = 'kfold'
##### Model ##### OLS, model = 'OLS', ##### Ridge, model = 'Ridge' ###### Lasso, model = 'Lasso'
def Regression(z, p = 3,lamb = 0, model = 'OLS', resampling = 'kfold', error ='MSE'):
    X = CreateDesignMatrix(x,y,p)

    if (model == 'OLS' or model == 'Ridge'):
        if model == 'OLS': lamb = 0
        betas = beta(X, z, lamb)
        #print("Betas: ", betas)
        print("polynomial: ", p, " Determinant: ", np.linalg.det(X.T.dot(X)))
        z_tilde = X.dot(betas)

        if resampling == 'none':
            Error = ErrorAnalysis(z, z_tilde, error)
            return Error, betas, z_tilde
        elif resampling == 'kfold':
            return Kfold(X,z, lamb)

    if model == 'Lasso':  # Here we used scikit learn
        if error != 'MSE': error = 'MSE' and print('only MSE available')
        model_lasso = skl.Lasso(alpha=lamb, fit_intercept=False, normalize=True,tol=0.01)

        if resampling == 'none':
            model_lasso.fit(X,z)
            betas = model_lasso.coef_
            z_tilde = X.dot(betas)
            Error = ErrorAnalysis(z,z_tilde, error)
            return Error, betas, z_tilde
        elif resampling == 'kfold':
            splits = 5
            k_fold = KFold(n_splits=splits, shuffle=True, random_state=seed)
            return np.mean(-(cross_val_score(model_lasso, X, z, scoring='neg_mean_squared_error', cv=k_fold)))
    return

def plotting(z, p = 3, lamb = 0, model = 'OLS', resampling = 'kfold', error = 'MSE', plot = 'polynomial'):
    complexity = np.arange(0, p + 1)
    if plot == 'polynomial':

        if error == 'MSE':
            if resampling == 'kfold':

                MSE_train_array = np.zeros(p+1)
                MSE_test_array = np.zeros(p+1)
                for p in range(p+1):
                    MSE_train_array[p], MSE_test_array[p] = Regression(z, p, lamb, model, resampling, error)[0:2]
                plt.xlabel('Model Complexity')
                plt.ylabel('Prediction Error (MSE)')
                plt.plot(complexity, MSE_train_array)
                plt.plot(complexity, MSE_test_array)
                plt.legend(['Train sample', 'Test sample'], loc='upper left')
                plt.show()
            elif resampling == 'none':

                MSE_array = np.zeros(p+1)
                for p in range(p+1):
                    MSE_array[p] = Regression(z, p, lamb, model, resampling = 'none', error = 'MSE')[0]
                plt.xlabel('Model Complexity')
                plt.ylabel('Prediction Error (MSE)')
                plt.plot(complexity, MSE_array)
                plt.legend(['Sample with no cross validation'], loc='upper left')
                plt.show()

        if error == 'R2':

            R2_array = np.zeros(p+1)
            for p in range(p+1):
                R2_array[p] = Regression(z, p, resampling = 'none', error = 'R2')[0]
            plt.xlabel('Model Complexity')
            plt.ylabel('R2')
            plt.plot(complexity, R2_array)
            plt.show()

    elif plot == 'lambdas': # only kfold
        maxlamb = np.log10(lamb)
        steps = 500
        lambdas = np.logspace(-3, maxlamb, steps)
        if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
        if model == 'Ridge':
            MSE_train_array = np.zeros((steps))
            MSE_test_array = np.zeros((steps))
            for i in range(steps):
                MSE_train_array[i], MSE_test_array[i] = Regression(z, p, lambdas[i], model, resampling, error)[0:2]

            plt.plot(np.log10(lambdas), MSE_train_array, label='$\lambda_{train}$')
            plt.plot(np.log10(lambdas), MSE_test_array, dashes=[6, 2], label='$\lambda_{test}$')
            plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
            plt.subplots_adjust(right=0.85)
            plt.xlabel('$log{\lambda}$')
            plt.ylabel('Prediction Error (MSE)')
            plt.show()
        if model == 'Lasso':
            MSE_test_array = np.zeros((steps))
            for i in range(len(lambdas)):
                MSE_test_array[i] = Regression(z, p, lambdas[i], model, resampling, error)

            plt.plot(np.log10(lambdas), MSE_test_array, dashes=[6, 2], label='$\lambda_{test}$')
            plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
            plt.subplots_adjust(right=0.85)
            plt.xlabel('$log{\lambda}$')
            plt.ylabel('Prediction Error (MSE)')
            plt.show()


    elif plot == 'lambdas polynomial':

        steps = 5
        maxlamb = np.log10(lamb)
        lambdas = np.logspace(-3, maxlamb, steps)
        lambdas = [round(l,5) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
        lambdas = [0,0.02,0.1,1,10]
        if resampling == 'kfold':

            if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
            elif model == 'Ridge':
                MSE_train_array = np.zeros((steps,p + 1))
                MSE_test_array = np.zeros((steps,p + 1))
                mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
                for i in range(len(lambdas)):
                    for p in range(p+1):
                        MSE_train_array[i][p], MSE_test_array[i][p] = Regression(z, p, lambdas[i], model, resampling, error)[0:2]

                    plt.plot(complexity, MSE_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
                    plt.plot(complexity, MSE_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
                plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
                plt.subplots_adjust(right=0.75)
                plt.xlabel('Model Complexity')
                plt.ylabel('Prediction Error (MSE)')
                plt.show()
            elif model == 'Lasso': # Only MSE_test. Cross_val_score(sklearn) only returns MSE_test.
                MSE_test_array = np.zeros((steps, p + 1))
                for i in range(len(lambdas)):
                    for p in range(p + 1):
                        MSE_test_array[i][p] = Regression(z, p, lambdas[i], model, resampling, error)

                    plt.plot(complexity, MSE_test_array[i], label='$\lambda$ = %s' % lambdas[i])
                plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.77), fancybox=True, shadow=True)
                plt.subplots_adjust(right=0.75)
                plt.xlabel('Model Complexity')
                plt.ylabel('Prediction Error (MSE)')
                plt.show()

        elif resampling == 'none':

            MSE_array = np.zeros((steps, p + 1))
            for i in range(len(lambdas)):
                for p in range(p+1):
                    MSE_array[i][p] = Regression(z, p, lambdas[i], model, resampling, error)[0]

                plt.plot(complexity, MSE_array[i], label = '$\lambda$ = %s'%lambdas[i] )
            plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.77), fancybox=True, shadow=True)
            plt.subplots_adjust(right=0.75)
            plt.xlabel('Model Complexity')
            plt.ylabel('Prediction Error (MSE)')
            plt.show()



""" def plotting(z, p = 3, lamb = 0, model = 'OLS', resampling = 'kfold', error = 'MSE', plot = 'polynomial') """

#X = CreateDesignMatrix(x,y,3)
#Confidence(X,z)
#beta, z_tilde = Regression(z)
#print(ErrorAnalysis(z,z_tilde))
#plotting(z,6,polyplot=True, error='R2')

#plotting(z,3,5, model = 'Ridge', plot = 'lambdas polynomial')
#plotting(z,15, 100000,model ='Lasso',resampling ='kfold', plot= 'lambdas')
plotting(z, 12, 10, model= 'OLS', resampling= 'kfold', plot ='polynomial')





"""
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
R = np.sqrt(x**2 + y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-100, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""