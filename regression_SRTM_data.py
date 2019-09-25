import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
import random
from scipy import stats
import platform
platform.architecture()

# Load the terrain
terrain = imread('SRTM_Oslo_Fjorden.tif')
#terrain = imread('SRTM_Norway_1.tif')
# Show the terrain

plt.figure()
plt.title('Terrain over Oslo Fjorden')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
#plt.show()



maxValue = np.amax(terrain)
random.seed(4000)

### Global variables
y_size = 1000
x_size = 1000
x = np.linspace(0, 1, x_size)
y = np.linspace(0, 1, y_size)
x, y = np.meshgrid(x, y)

# Pick a random cut of the map of certain size
col = random.randint(0,terrain.shape[1]-x_size)
row = random.randint(0,terrain.shape[0]-y_size)
cut = terrain[row:row+y_size, col:col+x_size]
z = cut

def CreateDesignMatrix(x,y,p):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(p)
    X = poly.fit_transform(np.concatenate((x, y), axis=1))

    return X

def beta(X,z,lamb = 0):
    return np.linalg.inv(X.T.dot(X) + lamb * np.identity(X.shape[1])).dot(X.T.dot(z))

def ErrorAnalysis(z1,z2, Error = 'MSE'):
    if Error == 'MSE':
        n = np.size(z2)
        return np.sum((z1 - z2) ** 2) / n
    elif Error == 'R2':
        return 1 - np.sum((z1 - z2) ** 2) / np.sum((z1 - np.mean(z2)) ** 2)
    return

def Confidence(X, z, betas = None, result = 'Errorbar', confidence = 0.95):

    if betas == None:
        z = z.reshape(-1, 1) # creates same result as np.ravel(z)
        betas = beta(X,z)

    SD_Array = []
    beta_number = []
    zScore = stats.norm.ppf(confidence)
    for i in range(len(betas)):
        Xs = X[:, i]
        Variance_beta = np.var(z) * (Xs.T.dot(Xs)) ** (-1)
        SD_beta = np.sqrt(Variance_beta)
        SD_Array = np.append(SD_Array, SD_beta)
        beta_number = np.append(beta_number, i)

    ####### error bars and plots
    betaArray = np.array(beta(X, z))

    zScore = stats.norm.ppf(0.95)
    sdArray = []
    x_value = []
    # for i in range(len(betaArray)):
    # Xs = X[:, i]
    sdArray = SDBeta(X, z)

    x_value = np.arange(len(betaArray))

    if result == 'Errorbar':
        yerr = 2 * (zScore * SD_Array / np.sqrt(X.size))         # xerr = 0.1
        plt.errorbar(x_value, betaArray, yerr, lw=1)
        plt.show()
    elif result == 'SD':
        return SD_Array

def Kfold(X, z, splits, lamb = 0):
    z = np.ravel(z)
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

        beta_train = beta(X_train, z_train,lamb)

        z_tilde = X_train.dot(beta_train)
        z_predict = X_test.dot(beta_train)

        MSE_train_i = ErrorAnalysis(z_tilde, z_train, Error='MSE')
        MSE_test_i = ErrorAnalysis(z_predict, z_test, Error='MSE')

        MSE_train = np.append(MSE_train, MSE_train_i)
        MSE_test = np.append(MSE_test, MSE_test_i)
    MSE_train_mean = np.mean(MSE_train,axis=0)
    MSE_test_mean = np.mean(MSE_test,axis=0)

    return MSE_test_mean, MSE_train_mean


### Regression ###

##### Resampling ##### No resampling, resampling = 'none' #####  Bootstrap, resampling = 'bootstrap' ##### k-fold, resampling = 'kfold'
##### Model ##### OLS, model = 'OLS', ##### Ridge, model = 'Ridge' ###### Lasso, model = 'Lasso'
def Regression(z, p = 2,lamb = 0, model = 'OLS', resampling = 'None', splits=5):
    X = CreateDesignMatrix(x,y,p)
    z = z.reshape(-1,1)

    if model == 'OLS':
        beta_OLS = beta(X, z)
        z_tilde = X.dot(beta_OLS)
        if resampling == 'None':
            return beta_OLS, z_tilde
        if resampling == 'bootstrap':
            return
        if resampling == 'kfold':
            MSE_test_mean, MSE_train_mean = Kfold(X, z, splits, lamb)
            return MSE_test_mean, MSE_train_mean, beta_OLS

    if model == 'Ridge':

        if resampling == 'None':
            I = np.identity(X.shape[1])
            beta_Ridge = beta(X,z,lamb)
            z_tilde = X.dot(beta_Ridge)
            return beta_ridge, z_tilde
        if resampling == 'bootstrap':
            return
    return



#X = CreateDesignMatrix(x,y,20)
#Confidence(X,z)
#beta, z_tilde = Regression(z)
#print(ErrorAnalysis(z,z_tilde))
print(Regression(cut,p=5,resampling='kfold')[0:2])





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
