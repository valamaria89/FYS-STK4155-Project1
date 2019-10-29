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
import matplotlib.pyplot as plt
from functools import partial

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
# print(beta_init)

#print(X_train_scaled)
#logrec = LogisticRegression()
#logrec.fit(XTrain,yTrain)
#print(logrec.coef_)

def sigmoid(X, beta):

    t = np.dot(X,beta)
    siggy = np.array(1 / (1 + np.exp(-t)),dtype=np.float128)
    return siggy

# def gradient_descent( X, y, beta, tol):
#     m = X.shape[0]
    # for iter in range(n_iter):
    #     gradient = (X.T.dot(y - sigmoid(X, beta)))
    #     beta -= eta*gradient
        # norm = np.linalg.norm(gradient)
        # if (norm < tol):
        #     print("GD donezzz")
        #     break
    #
    # return beta        #Is this code making sense?

class Weight:

    def __init__(self, X, y, beta, eta, iterations = 500):
        self.X = X
        self.y = y
        self.y_all = y
        self.X_all = X
        self.beta = beta
        self.eta = eta
        self.n = iterations
        self.e = 0

    def gradient_descent(self):
        gradient = -(self.X.T.dot(self.y - sigmoid(self.X, self.beta)))
        self.beta -= self.eta * gradient
        return self.beta

    def newtons_method(self):
        W = np.zeros(self.X.shape[0],self.X.shape[0])
        for i in range(len(self.X.shape[0])):
            np.fill_diagonal(W, sigmoid(self.X,self.beta) @ (1-sigmoid(self.X, self.beta)))
            self.beta = self.beta - np.linalg.pinv(self.X.T @ W @ self.X) @ (-self.X.T @ (self.y - sigmoid(self.X, self.beta)))
        return self.beta

    def learning_schedule(self, t):
        t0, t1 = 1, self.n
        return t0/(t+t1)
        
    def stochastick_gradient(self):

        M = 50
        m = int(self.X_all.shape[0]/M)
        shuffle_ind = np.arange(self.X_all.shape[0])
        np.random.shuffle(shuffle_ind)

        Xshuffled = np.zeros(self.X_all.shape)
        yshuffled = np.zeros(self.y_all.shape)
        for ind in range(self.X_all.shape[0]):

            Xshuffled[ind] = self.X_all[shuffle_ind[ind]]
            yshuffled[ind] = self.y_all[shuffle_ind[ind]]

   
        X_b = np.split(Xshuffled, m)
        y_b = np.split(yshuffled, m)
     
        for i in range(m):
            
            self.X = X_b[i]
            self.y = y_b[i]
            
            

            gradient = -(self.X.T.dot(self.y - sigmoid(self.X, self.beta)))
            #print(gradient)
            #print((self.gradient_descent()).shape)
            #print(self.eta)
            self.eta = self.learning_schedule((self.e*m+i)*1)
            self.beta -=  self.eta*gradient
            self.beta /=m
            #print(self.beta)
        #self.beta = np.linalg.norm(self.beta)
                
        return self.beta      

    def cost_function(self):
        return -np.sum(self.y * np.log(sigmoid(self.X, self.beta)) + (1 - self.y) * np.log(1 - sigmoid(self.X, self.beta))) / (len(self.y))

    def train(self, method):
        cost_all = np.array([])
        for i in range(self.n):
            self.e = i
            self.beta = method()
            #print(self.beta)
            # self.beta = self.gradient_descent()
            #print(self.beta)
            cost_current = self.cost_function()
            #print(cost_current)
            cost_all = np.append(cost_all, cost_current)
            #print(cost_all)
        return self.beta, cost_all 


# def new_weight(X, y, beta, eta):
#     gradient = -(X.T.dot(y - sigmoid(X, beta)))
#     beta -= eta * gradient
#     return beta, gradient
#
# def cost_function(X, y, beta):
#     Computes the cost function for all the training samples
    # cost_func = -np.sum(y*np.log(sigmoid(X,beta), dtype=np.float128) + (1-y)* np.log(1-sigmoid(X, beta),dtype=np.float128))/(len(y))
    # return cost_func

# def train(X, y, beta, eta, n_iter):
#     cost_all = np.array([])
#
#     for i in range(n_iter):
#         betas, gradient = new_weight(X, y, beta, eta)
        # w = Weight(X,y,beta,eta)
        # betas, gradient = w.gradient_descent()
        # norm = np.linalg.norm(gradient)
        # cost_current = cost_function(X, y, betas)
        # cost_all = np.append(cost_all,cost_current)
    #
    # return betas, cost_all



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


#final_betas, cost_all = train(X_train_scaled, y_train, beta_init, eta, n_iter, 1e-04)
# print(classification(X_train_scaled,final_betas))
#epoch = np.arange(len(cost_all))
#prob, y_pred, accuracy = classification(X_test_scaled, final_betas, y_test)
#false_pos, true_pos = roc_curve(y_test, prob)[0:2]



def Plots(costfunc_plot, LR_plot, ROC_plot, StockG):
    w = Weight(X_train_scaled,y_train,beta_init,eta, 100)
    if (costfunc_plot == 1):
        _,cost_all = w.train( w.gradient_descent)
        epoch = np.arange(len(cost_all))
        plt.plot(epoch, cost_all)
        plt.show()

    elif (LR_plot == 1):
        final_betas,_ = w.train(w.gradient_descent)
        prob_train = classification(X_train_scaled, final_betas)[0]
        x_sigmoid = np.dot(X_train_scaled, final_betas)
        plt.scatter(x_sigmoid, prob_train)
        plt.show()

    elif (ROC_plot==1):
        final_betas,_ = w.train(w.gradient_descent)
        prob, y_pred = classification(X_test_scaled, final_betas, y_test)[0:2]
        false_pos, true_pos = roc_curve(y_test, prob)[0:2]
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(false_pos, true_pos)
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.title("ROC curve gradient descent")
        plt.show()

    elif (StockG == 1):
        _,cost_all = w.train(w.stochastick_gradient)
        epoch = np.arange(len(cost_all))
        plt.plot(epoch, cost_all)
        plt.show()




Plots(0, 0, 0, 1)
#ConfMatrix = confusion_matrix(y, y_pred)

# plt.scatter(x_sigmoid,prob)
# plt.show()
# plt.plot(epoch,cost_all)
# plt.show()

# print(train(X, y, beta_init, eta, n_iter, 1e-04))
#parameters = best_fit(XTrain, yTrain, beta)       
# np.column_stack((prob,default))
