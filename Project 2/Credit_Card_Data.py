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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

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

eta = 1e-08 #Learning rate 
n_iter = 100
#XTrain = np.c_[np.ones((XTrain.shape[0], 1)), XTrain]
#yTrain = yTrain[:, np.newaxis]
beta =np.random.randn(X_train.shape[1],1)*0.1
#print(beta.shape)
#print(yTrain.shape)


#logrec = LogisticRegression()
#logrec.fit(XTrain,yTrain)
#print(logrec.coef_)

def sigmoid(X, beta):
    #print(X.shape)
    #print(beta.shape)
    t = np.dot(X,beta)
    return 1/(1+np.exp(-t))

def gradient_descent( X, y, beta, tol): 
    #m = X.shape[0] 
    for iter in range(n_iter):
        gradient = (X.T.dot(y - sigmoid(X, beta)))
        beta -= eta*gradient
        norm = np.linalg.norm(gradient)
        #print(beta[10])
        if (norm < tol):
            print("GD donezzz")
            break
        
    return beta        #Is this code making sense? 

beta = gradient_descent(X_train, y_train, beta, 1e-04)
print(beta)

def cost_function(X, y, beta):
    #Computes the cost function for all the training samples
    #m = X.shape[0]
    cost_func = -np.sum(y*np.log(sigmoid(X,beta)) + (1-y)* np.log(1-sigmoid(X, beta)))
    return cost_func



#parameters = best_fit(XTrain, yTrain, beta)       


