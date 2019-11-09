import pandas as pd
import os
import numpy as np
import xlrd
import sys
from scipy.optimize import fmin_tnc
np.set_printoptions(threshold=sys.maxsize)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from functools import partial
import scikitplot as skplt

# Trying to set the seed

import random

seed = 42
random.seed(seed)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
# filename = cwd + '/test.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

"""df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_0 ==0) &
                (df.PAY_2 == 0) &
                (df.PAY_3 == 0) &
                (df.PAY_4 == 0) &
                (df.PAY_5 == 0) &
                (df.PAY_6 == 0)].index)
"""


df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE']=np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
df['MARRIAGE'].unique()

#df.drop('ID', axis = 1, inplace =True)

#df.info()

#categorical_subset = pd.get_dummies(categorical_subset[categorical_subset.columns.drop("defaultPaymentNextMonth")])
# Features and targets
#X = df.drop('defaultPaymentNextMonth', axis = 1, inplace = False)
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Frequency default
yes = df.defaultPaymentNextMonth.sum()
no = len(df)-yes

#precentage 
yes_perc = round(yes/len(df)*100, 1)
no_perc = round(no/len(df)*100, 1)

print("Default: ", yes_perc)
print("Non-Default: ", no_perc)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Categorical variables to one-hot's
"""onehotencoder = OneHotEncoder(categories="auto") #What does auto do ? 


X = ColumnTransformer(
    [("", onehotencoder, [3]),],
    remainder="passthrough"
).fit_transform(X) #Why do we only do this on the marriage column? 



y.shape

# Train-test split
trainingShare = 0.5
seed  = 1000
XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)


# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

# One-hot's of the target vector
Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest) # Why do you do this? 
"""

# Remove instances with zeros only for past bill statements or paid amounts
'''
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0) &
                (df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)
'''






############ Logistic Regression with Gradient Descent ##################

eta = 0.0001 #Learning rate

#n_iter = 450
#XTrain = np.c_[np.ones((XTrain.shape[0], 1)), XTrain]
#yTrain = yTrain[:, np.newaxis]
beta_init = np.random.randn(X_train.shape[1],1)
#print(beta_init.shape)

#print(X_train_scaled)
#logrec = LogisticRegression()
#logrec.fit(XTrain,yTrain)
#print(logrec.coef_)

def sigmoid(X, beta):

    t = X @ beta
    siggy = np.array(1 / (1 + np.exp(-t)),dtype=np.float128)
    return siggy

class Weight:

    def __init__(self, X, y, beta, eta, iterations = 500):
        self.X = X
        self.y = y
        self.y_all = y
        self.X_all = X
        self.beta = beta
        self.eta = eta
        self.iterations = iterations
        self.epoch = 0
        #self.cost = np.array([])

    def gradient_descent(self):
        gradient = -(self.X.T.dot(self.y - sigmoid(self.X, self.beta)))
        self.beta -= self.eta * gradient
        self.cost = self.cost_function()
        return self.beta

    def newtons_method(self):
        W = np.zeros((self.X.shape[0],self.X.shape[0]))
        for i in range(self.X.shape[0]):
            W[i][i] = (sigmoid(self.X[i],self.beta)) @ (1-sigmoid(self.X[i], self.beta))
        self.beta = self.beta - np.linalg.pinv(self.X.T @ W @ self.X) @ (-self.X.T @ (self.y - sigmoid(self.X, self.beta)))
        self.cost = self.cost_function()
        return self.beta

    def learning_schedule(self, t):
        t0, t1 = 120, 12000#self.iterations
        return t0/(t+t1)

    def shuffle(self):
        shuffle_ind = np.arange(self.X_all.shape[0])
        
        np.random.shuffle(shuffle_ind)

        Xshuffled = np.zeros(self.X_all.shape)
        yshuffled = np.zeros(self.y_all.shape)
        for ind in range(self.X_all.shape[0]):
            Xshuffled[ind] = self.X_all[shuffle_ind[ind]]
            yshuffled[ind] = self.y_all[shuffle_ind[ind]]
        return Xshuffled, yshuffled
        
    def mini_batch_gradient_descent(self):
        M = 32
        m = int(self.X_all.shape[0]/M)
        
        Xshuffled, yshuffled = self.shuffle()
        X_b = np.split(Xshuffled, m)
        y_b = np.split(yshuffled, m)

        
        self.cost = np.array([])
        for i in range(m):
            self.X = X_b[i]
            self.y = y_b[i]
            
            gradient = -(self.X.T @ (self.y - sigmoid(self.X, self.beta)))
            self.eta = self.learning_schedule((self.epoch*m+i)*1)
            self.beta -=  (self.eta*gradient)
            self.cost = np.append(self.cost, self.cost_function())

        return self.beta

    def stochastic_gradient_descent(self):
        m = len(self.y_all)
        #random.seed(seed)
        Xshuffled, yshuffled = self.shuffle()
        
        self.cost = np.array([])
        for i in range(m):

            self.X = Xshuffled[i:i+1,:]
            self.y = yshuffled[i:i+1,:]
            gradient = -(self.X.T @ (self.y - sigmoid(self.X, self.beta)))
            self.eta = self.learning_schedule((self.epoch*m+i)*1)
            self.beta -=  self.eta*gradient
            self.cost = np.append(self.cost, self.cost_function())
        return self.beta

    def stochastic_gradient_descent_Skl(self):
        m = len(self.y_all)
        self.cost = np.array([])
        # self.X, self.y = self.shuffle()
        clf = SGDClassifier(loss="log", penalty="l2", max_iter=self.iterations, shuffle=True, random_state=seed)
        clf.fit(self.X_all, self.y_all.ravel(), coef_init=self.beta)
        self.beta = (clf.coef_).T
        return self.beta, clf.predict_proba

    def cost_function(self):
        return -np.sum(self.y * np.log(sigmoid(self.X, self.beta)) + (1 - self.y) * np.log(1 - sigmoid(self.X, self.beta))) / (len(self.y))

    def train(self, method):
        self.cost_all = np.array([])
        if 'Skl' in str(method):
            self.iterations = 1
        for i in range(self.iterations):
            self.epoch = i
            self.beta = method()
            self.cost_all = np.append(self.cost_all, self.cost)
        return self.beta, self.cost_all


def classification(X, betas, y_test=[0]):
    prob = sigmoid(X,betas)
    tot = 0
    y_pred = np.zeros(X.shape[0])
    for i in range(len(y_pred)):
        if prob[i] >= .5:
            y_pred[i] =1
        else:
            y_pred[i] = 0
        if ( np.sum(y_test) != 0 and y_pred[i] == y_test[i]):
            tot += 1

    return prob, y_pred, tot/len(y_pred)

def logistic_regression_SKL(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=seed, solver='lbfgs',multi_class = 'ovr').fit(X_train, y_train.ravel())
    prob_LogReg_Skl = clf.predict_proba(X_test)[:,1:2]
    print(prob_LogReg_Skl)
    false_pos_LogReg_Skl, true_pos_LogReg_Skl = roc_curve(y_test, prob_LogReg_Skl)[0:2]
    print("Area under curve LogReg_skl: ", auc(false_pos_LogReg_Skl, true_pos_LogReg_Skl))
    plt.show()

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(false_pos_LogReg_Skl, true_pos_LogReg_Skl, label="LogReg")
    plt.legend()
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title("ROC curve")
    plt.show()
    return


#final_betas, cost_all = train(X_train_scaled, y_train, beta_init, eta, n_iter, 1e-04)
# print(classification(X_train_scaled,final_betas))
#epoch = np.arange(len(cost_all))
#prob, y_pred, accuracy = classification(X_test_scaled, final_betas, y_test)
#false_pos, true_pos = roc_curve(y_test, prob)[0:2]



def Plots(ROC_plot, GD_plot, MB_GD_plot, Stoch_GD_plot, Newton_plot, Scatter_GD_plot):
    w = Weight(X_train_scaled,y_train,beta_init,eta, 5)
    if (ROC_plot==1):
        # final_betas_grad,_ = w.train(w.gradient_descent)
        # prob_grad, y_pred_grad = classification(X_test_scaled, final_betas_grad, y_test)[0:2]
        # false_pos_grad, true_pos_grad = roc_curve(y_test, prob_grad)[0:2]
        # print("Area under curve gradient: ", auc(false_pos_grad, true_pos_grad))
        #
        final_betas_MB,_ = w.train(w.mini_batch_gradient_descent)
        prob_MB, y_pred_MB = classification(X_test_scaled,final_betas_MB, y_test)[0:2]
        false_pos_MB, true_pos_MB = roc_curve(y_test, prob_MB)[0:2]
        print("Area under curve MB: ", auc(false_pos_MB, true_pos_MB))

        final_betas_ST,_ = w.train(w.stochastic_gradient_descent)
        prob_ST, y_pred_ST = classification(X_test_scaled,final_betas_ST, y_test)[0:2]
        false_pos_ST, true_pos_ST = roc_curve(y_test, prob_ST)[0:2]
        print("Area under curve ST: ", auc(false_pos_ST, true_pos_ST))

        # final_betas_ST_Skl,_ = w.train(w.stochastic_gradient_descent_Skl)
        # prob_ST_Skl, y_pred_ST_Skl = classification(X_test_scaled,final_betas_ST_Skl, y_test)[0:2]
        # false_pos_ST_Skl, true_pos_ST_Skl = roc_curve(y_test, prob_ST_Skl)[0:2]
        # print("Area under curve ST_skl: ", auc(false_pos_ST_Skl, true_pos_ST_Skl))

        # final_betas_Newton,_ = w.train(w.newtons_method)
        # prob_Newton, y_pred_Newton = classification(X_test_scaled,final_betas_Newton, y_test)[0:2]
        # false_pos_Newton, true_pos_Newton = roc_curve(y_test, prob_Newton)[0:2]
        # print("Area under curve Newton: ", auc(false_pos_Newton, true_pos_Newton))

        plt.plot([0, 1], [0, 1], "k--")
        # plt.plot(false_pos_grad, true_pos_grad,label="Gradient")
        plt.plot(false_pos_ST, true_pos_ST, label="Stoch")
        # plt.plot(false_pos_ST_Skl, true_pos_ST_Skl, label="Stoch_Skl")
        plt.plot(false_pos_MB, true_pos_MB, label="Mini")
        # plt.plot(false_pos_Newton, true_pos_Newton, label="Newton")
        plt.legend()
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.title("ROC curve")
        # plt.show()

    if (GD_plot == 1):
        _, cost_all = w.train(w.gradient_descent)
        epoch = np.arange(len(cost_all))

        plt.plot(epoch, cost_all)
        plt.show()

    if (MB_GD_plot == 1):
        _,cost_all = w.train(w.mini_batch_gradient_descent)
        batch = np.arange(len(cost_all))
        
        plt.plot(batch, cost_all)
        plt.show()

    if (Stoch_GD_plot == 1):
        _,cost_all = w.train(w.stochastic_gradient_descent)
        batch = np.arange(len(cost_all))
        
        plt.plot(batch, cost_all)
        plt.show()

    if (Newton_plot == 1):
        _,cost_all = w.train(w.newtons_method)
        epochs = np.arange(len(cost_all))

        plt.plot(epochs, cost_all)
        plt.show()

    if (Scatter_GD_plot == 1):
        final_betas,_ = w.train(w.gradient_descent)
        prob_train = classification(X_train_scaled, final_betas)[0]
        x_sigmoid = np.dot(X_train_scaled, final_betas)
        plt.scatter(x_sigmoid, prob_train)
        plt.show()


# Plots(1, 0, 0, 0, 0, 0)
logistic_regression_SKL(X_train_scaled, X_test_scaled, y_train, y_test)

# Plots(0, 0, 1, 0, 0)
#ConfMatrix = confusion_matrix(y, y_pred)

# plt.scatter(x_sigmoid,prob)
# plt.show()
# plt.plot(epoch,cost_all)
# plt.show()

# print(train(X, y, beta_init, eta, n_iter, 1e-04))
#parameters = best_fit(XTrain, yTrain, beta)       
# np.column_stack((prob,default))
