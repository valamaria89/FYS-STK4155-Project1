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
from matplotlib.patches import Rectangle
from PIL import Image

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.linear_model as skl

colors = ['#1f77b4', '#1f77b4', '#aec7e8','#aec7e8','#ff7f0e','#ff7f0e','#d62728','#d62728','#2ca02c','#2ca02c','#98df8a','#98df8a','#ffbb78','#ffbb78']
# Load the terrain
terrain = imread('SRTM_Oslo_Fjorden.tif')

# Show the terrain

# plt.figure()
# currentAxis = plt.gca()
# currentAxis.add_patch(Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none'))
# plt.title('Terrain over Oslo Fjorden')
# plt.imshow(terrain, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()



seed = 3001
maxValue = np.amax(terrain)
terrain_shape = terrain.shape
random.seed(seed)

### Global variables
def MapMaker(terrain, x_size, y_size):
    x = np.linspace(0, 1, x_size)
    y = np.linspace(0, 1, y_size)
    x, y = np.meshgrid(x, y)

    # Pick a random cut of the map of certain size
    col = random.randint(0,terrain.shape[1]-x_size)
    row = random.randint(0,terrain.shape[0]-y_size)
    cut = terrain[row:row+y_size, col:col+x_size]
    z = cut
    z = z.reshape(-1,1)
    return z, x, y, col, row
z,x_array,y_array = MapMaker(terrain, 40, 40)[0:3]



def CreateDesignMatrix(x,y,p):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(p)
    X = poly.fit_transform(np.concatenate((x, y), axis=1))

    return X

def beta(X,z,lamb = 0):
    z = np.ravel(z)
    return np.linalg.pinv(X.T.dot(X) + lamb * np.identity(X.shape[1])).dot(X.T.dot(z))

def Variance_Bias_Analysis(X, z, z_tilde, lamb = 0, betas = None, result = 'Errorbar', confidence = 0.95):

    if betas.all() == None:
        z = z.reshape(-1, 1)
        betas = beta(X,z, lamb)
    variance_z = (np.sum(z-z_tilde)**2)/(len(z)-len(betas)-1)
    Variance_Beta = np.diag(variance_z * np.linalg.pinv(X.T.dot(X)))
    SD_array = np.sqrt(Variance_Beta)
    zScore = stats.norm.ppf(confidence)
    beta_n = np.arange(len(betas))

    if result == 'Errorbar':
        yerr = zScore * SD_array
        plt.errorbar(beta_n, betas, yerr, lw=1,linestyle=':',marker='o')
        plt.show()
    elif result == 'SD': return SD_array
    elif result == 'Variance': return Variance_Beta

def Error_Analysis(z1,z2, error = 'MSE'):  # e.g z1 = z_train, z2 = ztilde or z1 = z_test, z2 = z_predict
    z1 = np.ravel(z1)
    z2 = np.ravel(z2)
    if error == 'MSE':
        n = np.size(z2)
        return np.sum((z1 - z2) ** 2) / n
    elif error == 'R2':
        return 1 - np.sum((z1 - z2) ** 2) / np.sum((z1 - np.mean(z1)) ** 2)
    elif error == 'MSE and R2':
        n = np.size(z2)
        MSE = np.sum((z1 - z2) ** 2) / n
        R2 = 1 - np.sum((z1 - z2) ** 2) / np.sum((z1 - np.mean(z1)) ** 2)
        return MSE, R2
    return


def Kfold(X, z, lamb = 0, model_skl =None, intercept = True):

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
    R2_train = np.zeros((splits))
    R2_test = np.zeros((splits))
    for i in range(splits):

        X_train = X_k
        X_train = np.delete(X_train, i, 0)
        X_train = np.concatenate(X_train)

        z_train = z_k
        z_train = np.delete(z_train, i, 0)
        z_train = np.ravel(z_train)

        X_test = X_k[i]
        z_test = z_k[i]

        if (model_skl != None):  ### for lasso
            if(intercept == False and X_train.shape[1]>1):
                X_no_train = X_train[:, 1:]
                model_skl.fit(X_no_train, z_train)
                beta_train = model_skl.coef_
                beta_train = np.insert(beta_train, 0, model_skl.intercept_)

            else:
                model_skl.fit(X_train, z_train)
                beta_train = model_skl.coef_
        else: beta_train = beta(X_train, z_train, lamb)

        z_tilde = X_train.dot(beta_train)
        z_predict = X_test.dot(beta_train)

        MSE_train[i], R2_train[i] = Error_Analysis(z_train, z_tilde, error='MSE and R2')
        MSE_test[i], R2_test[i] = Error_Analysis(z_test, z_predict, error='MSE and R2')

    MSE_train_mean = np.mean(MSE_train,axis=0)
    MSE_test_mean = np.mean(MSE_test,axis=0)
    R2_train_mean = np.mean(R2_train,axis=0)
    R2_test_mean = np.mean(R2_test,axis=0)

    return MSE_train_mean, MSE_test_mean, R2_train_mean, R2_test_mean

### Regression ###

def Regression(z, p = 3,lamb = 0, model = 'OLS', resampling = 'kfold', error ='MSE', x = x_array, y = y_array, intercept = True):
    X = CreateDesignMatrix(x,y,p)

    if (model == 'OLS' or model == 'Ridge'):
        if model == 'OLS': lamb = 0

        betas = beta(X, z, lamb)

        #print("polynomial: ", p, " Determinant: ", np.linalg.det(X.T.dot(X)))
        z_tilde = X.dot(betas)

        if resampling == 'none':
            MSE, R2 = Error_Analysis(z, z_tilde, "MSE and R2")
            var_betas = Variance_Bias_Analysis(X, z, z_tilde, lamb, result='Variance', betas=betas)
            return MSE, R2, var_betas, betas, z_tilde
        elif resampling == 'kfold':
            return Kfold(X,z, lamb, intercept=intercept)

    if model == 'Lasso':  # Here we used scikit learn
        if intercept == False: fitinter = True
        else: fitinter = False
        model_lasso = skl.Lasso(alpha=lamb, fit_intercept=fitinter, normalize=True,tol=6000)

        if resampling == 'none':
            model_lasso.fit(X, z)
            if intercept == False:
                X_no = X[:, 1:]
                model_lasso.fit(X_no, z)
                betas = model_lasso.coef_
                betas = np.insert(betas, 0, model_lasso.intercept_)
            else: betas = model_lasso.coef_
            z_tilde = X.dot(betas)
            MSE, R2 = Error_Analysis(z, z_tilde, "MSE and R2")
            var_betas = Variance_Bias_Analysis(X, z, z_tilde, lamb, result='Variance', betas=betas)
            return MSE, R2, var_betas, betas, z_tilde
        elif resampling == 'kfold':
            return Kfold(X, z, lamb, model_lasso,intercept=intercept)
    return

def plotting(z, p = 3, lamb = 0, model = 'OLS', resampling = 'kfold', error = 'MSE', plot = 'polynomial', find_optimal = False, intercept = True, x = x_array, y = y_array):
    complexity = np.arange(0, p + 1)
    if error == 'MSE': j = 2
    elif error == 'R2' : j = 4

########### plots MSE with respect to polynomial degree ########
    if plot == 'polynomial':
        if resampling == 'kfold':

            error_train_array = np.zeros(p+1)
            error_test_array = np.zeros(p+1)
            poly = p+1
            for p in range(poly):
                error_train_array[p], error_test_array[p] = Regression(z, p, lamb, model, resampling, error, intercept=intercept)[j-2:j]
                if find_optimal == True: print(p)
            if find_optimal == True:
                ind = np.argmin(error_test_array)
                print("Best polynomial",complexity[ind])
                return complexity[ind]
            plt.xlabel('Model Complexity')
            plt.ylabel('Prediction Error (%s)'%error)
            plt.plot(complexity, error_train_array)
            plt.plot(complexity, error_test_array)
            plt.legend(['Train sample', 'Test sample'], loc='upper left')
            plt.suptitle(model, x=0.51, y=0.98, fontsize=10, fontweight='bold')
            plt.show()


        elif resampling == 'none':

            error_array = np.zeros(p+1)
            for p in range(p+1):
                error_array[p] = Regression(z, p, lamb, model, resampling, error, intercept=intercept)[int(j/2)-1]
            plt.xlabel('Model Complexity')
            plt.ylabel('Prediction Error (%s)'%error)
            plt.plot(complexity, error_array)
            plt.legend(['Sample with no cross validation'], loc='upper left')
            plt.suptitle(model, x=0.51, y=0.98, fontsize=10, fontweight='bold')
            plt.show()

###### plot lambdas gives MSE with respect to lambda degree, adjust minimum value inside the function. Default is -16 #####

    elif plot == 'lambdas': # only kfold
        if resampling != 'kfold':
            print('only k-fold')
            resampling = 'kfold'
        maxlamb = np.log10(lamb)
        steps = 50
        lambdas = np.logspace(-16, maxlamb, steps)
        if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
        else:
            error_train_array = np.zeros((steps))
            error_test_array = np.zeros((steps))
            for i in range(steps):
                error_train_array[i], error_test_array[i] = Regression(z, p, lambdas[i], model, resampling, error, intercept=intercept, x = x, y = y)[j-2:j]

            if find_optimal == True:
                ind = np.argmin(error_test_array)
                return(lambdas[ind], error_test_array[ind])

            plt.plot(np.log10(lambdas), error_train_array, label='$\lambda_{train}$')
            plt.plot(np.log10(lambdas), error_test_array, dashes=[6, 2], label='$\lambda_{test}$')
            plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
            plt.subplots_adjust(right=0.85)
            plt.xlabel('$log{\lambda}$')
            plt.ylabel('Prediction Error (%s)'%error)
            plt.suptitle(model, x=0.51, y=0.98, fontsize=10, fontweight='bold')
            plt.show()


##### plots different graphs of lambda with resepct to polynomial degree and MSE, plots both test and train###
    elif plot == 'lambdas polynomial':

        steps = 6
        maxlamb = np.log10(lamb)
        lambdas = np.logspace(-3, maxlamb, steps)
        lambdas = [round(l,6) for l in lambdas] # just removing some non-crucial decimals to make the graph labels fit the plot
        #lambdas = [0,0.0001,0.001,0.01,0.1,1]
        if resampling == 'kfold':

            if model == 'OLS': print("Can't iterate over lambdas. Model='OLS' means lambda=0. Use plot='polynomial' or model='Ridge' or 'Lasso' instead")
            else:
                error_train_array = np.zeros((steps,p + 1))
                error_test_array = np.zeros((steps,p + 1))
                mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
                for i in range(len(lambdas)):
                    for p in range(p+1):
                        error_train_array[i][p], error_test_array[i][p] = Regression(z, p, lambdas[i], model, resampling, error, intercept=intercept)[j-2:j]

                    plt.plot(complexity, error_train_array[i], label = '$\lambda_{train}$ = %s'%lambdas[i] )
                    plt.plot(complexity, error_test_array[i], dashes=[6, 2], label='$\lambda_{test}$ = %s'%lambdas[i])
                plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.6), fancybox=True, shadow=True)
                plt.subplots_adjust(right=0.75)
                plt.xlabel('Model Complexity')
                plt.ylabel('Prediction Error (%s)'%error)
                plt.suptitle(model, x=0.51, y=0.98, fontsize=10, fontweight='bold')
                plt.show()

        elif resampling == 'none':

            error_array = np.zeros((steps, p + 1))
            for i in range(len(lambdas)):
                for p in range(p+1):
                    error_array[i][p] = Regression(z, p, lambdas[i], model, resampling, error, intercept=intercept)[int(j/2)-1]

                plt.plot(complexity, error_array[i], label = '$\lambda$ = %s'%lambdas[i] )
            plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.77), fancybox=True, shadow=True)
            plt.subplots_adjust(right=0.75)
            plt.xlabel('Model Complexity')
            plt.ylabel('Prediction Error (%s)'%error)
            plt.show()



"""def plotting(z, p = 3, lamb = 0, model = 'OLS', resampling = 'kfold', error = 'MSE', plot = 'polynomial')"""


# plotting(z, 2, 0.01, model= 'Ridge', resampling= 'kfold', plot ='polynomial')
# plotting(z,10,100,model='Ridge', resampling='kfold', plot='lambdas')
# plotting(z,5,1, model="Ridge",resampling='kfold',plot='lambdas polynomial')


#### best_fit ### finds the best function with best MSE given lambda and poly ####
def best_fit(z, maxpoly = 50, maxlamb = 1, model = 'Ridge', intercept = True, x=x_array, y=y_array):
    if model == 'OLS':
        poptimal = plotting(z, maxpoly, 0, model="OLS", resampling='kfold', plot='polynomial', find_optimal=True, x=x, y=y)
        return poptimal
    else:
        opt_lambda, opt_error = plotting(z, 0, maxlamb, model, plot='lambdas', find_optimal=True, intercept=intercept, x=x, y=y)
        poptimal = 0
        print("\npolynomial", 0)
        print("optimal MSE ", opt_error)
        print("optimal lambda ", opt_lambda)
        for p in range(1,maxpoly+1):
            opt_lambda_i, opt_error_i = plotting(z, p, maxlamb, model, plot= 'lambdas', find_optimal=True, intercept=intercept, x=x, y=y)
            print("Checking polynomia ", p)
            if opt_error_i < opt_error:
                opt_error = opt_error_i
                opt_lambda = opt_lambda_i
                poptimal = p
                print("current optimal MSE ", opt_error)
                print("current optimal lambda", opt_lambda)
                print("")
        print("\nfinal optimal poly ", poptimal)
        print("final optimal MSE ", opt_error)
        print("final optimal lambda \n", opt_lambda)
        return poptimal, opt_lambda


#### This function plots all the terrain plots, prints out to text, and runs the best fit function if bestfit=True ####
colors2 = ['#5D8AA8','#E32636','#9966CC','#008000','#00FFFF','#FDEE00']
def surface_plotting(terrain, maxpoly, lamb=0, model='OLS', num_cuts=3, xsize = 20, ysize = 20, title = '',intercept = True, start_cut = 0, bestfit =  True, textnum=''):
    model_text = model + textnum
    with open("%s values for terrain data.txt"%model_text, "a") as f:
        if bestfit == False: print("**** %s ****"%model,"\npolynomial = %s"%p,"\nlambda = %s"%lamb, file=f)
        else: print("**** %s ****"%model, file=f)
    fig, axs = plt.subplots(num_cuts, 2, subplot_kw={'projection': '3d'})
    plt.subplots_adjust(hspace=0.8)
    fig.suptitle(title,x= 0.51,y =0.98, fontsize=12, fontweight='bold')
    plt.figure()
    for num_cut in range(start_cut,num_cuts+start_cut):
        terrain_cut,x_array_plt,y_array_plt, colbox, rowbox = MapMaker(terrain, xsize, ysize)

        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((colbox, rowbox), xsize, ysize, linewidth=2, edgecolor=colors2[num_cut], facecolor='none'))

        if bestfit == True:
            if model == 'OLS':
                p = best_fit(terrain_cut, x = x_array_plt, y = y_array_plt, model=model, maxpoly = maxpoly)
                lamb = 0
            else:
                p, lamb = best_fit(terrain_cut, x = x_array_plt, y = y_array_plt, model=model, maxpoly = maxpoly)
        with open("%s values for terrain data.txt" % model_text, "a") as f:
            print("\nTerrain cut %s\n" % (num_cut + 1), file=f)
            print("\npolynomial = %s" % p, "\nlambda = %s\n" % lamb, file=f)
        MSE, R2, var_betas, betas, fitted_cut = Regression(terrain_cut,p,lamb,model,resampling='none', x = x_array_plt, y = y_array_plt, intercept=intercept)

        rows = np.arange(ysize)
        cols = np.arange(xsize)

        [X, Y] = np.meshgrid(cols, rows)

        terrain_cut = np.reshape(terrain_cut, (ysize, xsize))
        fitted_cut = np.reshape(fitted_cut, (ysize, xsize))
        num_ax = num_cut*2

        if start_cut != 0:
            num_ax = num_ax - 2*start_cut

        ax = fig.axes[num_ax]
        ax.plot_surface(X, Y, fitted_cut, cmap=cm.viridis, linewidth=0)
        ax.set_title('Fitted terrain cut = %s' % (num_cut+1))

        ax = fig.axes[num_ax+1]
        ax.plot_surface(X, Y, terrain_cut, cmap=cm.viridis, linewidth=0)
        ax.set_title('Terrain cut = %s' % (num_cut+1))

        with open("%s values for terrain data.txt"%model_text, "a") as f:
            print("\nParameters with uncertainties", file=f)
            for i in range(len(betas)):
                print(round(betas[i],6), " +/- ", round(np.sqrt(var_betas[i]),6), file=f)
            MSE_train_mean, MSE_test_mean, R2_train_mean, R2_test_mean = Regression(terrain_cut, p, lamb, model, resampling='kfold', x=x_array_plt, y=y_array_plt, intercept=intercept)
            print("\nMSE whole data set = ", MSE, file=f)
            print("R2 whole data set = ", R2, file=f)
            print("\nResampling:",file=f)
            print("\nMean MSE on train data = ", MSE_train_mean, file=f)
            print("Mean MSE on test data = ", MSE_test_mean, file=f)
            print("\nMean R2 on train data = ", R2_train_mean, file=f)
            print("Mean R2 on test data = ", R2_test_mean, file=f)
            print("\n***********************************************", file=f)

    plt.title('Terrain over Oslo Fjorden')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


#### These plots the terrain cuts of min MSE
# surface_plotting(terrain,30,model='Lasso', title ='OLS regression', xsize=40, ysize=40, num_cuts=3, intercept=True, start_cut=0, textnum='1')
# surface_plotting(terrain,30,model='Lasso', title ='OLS regression', xsize=40, ysize=40, num_cuts=3, intercept=True, start_cut=3, textnum='2')


