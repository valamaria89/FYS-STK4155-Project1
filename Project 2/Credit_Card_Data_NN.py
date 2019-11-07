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

import tensorflow as tf
import seaborn as sns
import os

from NeuralNetworkTensorFlow import NeuralNetworkTensorflow as NNT
from NeuralNetwork import NeuralNetwork as NN

tf.compat.v1.disable_eager_execution()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE']=np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
df['MARRIAGE'].unique()

# plt.matshow(df.corr())
# plt.show()

X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

y_train_onehot, y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

def Correlation(df):
    del df['defaultPaymentNextMonth']
    sns.set()
    # rs = np.random.RandomState(0)
    # df = pd.DataFrame(rs.rand(10, 10))
    # corr = df.corr()

    # plt.matshow(corr)
    # plt.show()
    # corr.show()
    fig, ax = plt.subplots(figsize=(20, 15))
    corr = df.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
    ax = sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True, fmt='.2f', annot_kws={"size": 7})
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'

    );
    ax.set_ylim(df.shape[1], 0)
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

Correlation(df)

def NN():
    epochs = 20
    batch_size = 10
    eta = 15
    lmbd = 0.01
    n_hidden_neurons = 30
    n_categories = 2

    DNN = NN(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                        cost_grad = 'crossentropy', activation = 'sigmoid', activation_out='ELU')
    DNN.train()
    test_predict = DNN.predict(X_test_scaled)
    test_predict1 = DNN.predict_probabilities(X_test_scaled)[:,1:2]
    #
    # accuracy score from scikit library
    #print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))
    #
    # def accuracy_score_numpy(Y_test, Y_pred):
    #     return np.sum(Y_test == Y_pred) / len(Y_test)

    false_pos, true_pos = roc_curve(y_test, test_predict1)[0:2]
    print("Area under curve ST: ", auc(false_pos, true_pos))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(false_pos, true_pos)
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title("ROC curve gradient descent")
    plt.show()

def NN_tf():
    epochs = 2
    batch_size = 100
    eta = 0.01
    lmbd = 0.01
    n_neurons_layer1 = 10
    n_neurons_layer2 = 5
    n_categories = 2
    DNN = NNT(X_train_scaled, y_train_onehot, X_test_scaled, y_test_onehot,
              n_neurons_layer1=n_neurons_layer1,
              n_neurons_layer2=n_neurons_layer2,
              n_categories=n_categories, epochs=epochs, batch_size=batch_size, eta=eta,
              lmbd=lmbd)
    DNN.fit()
    print("Test accuracy: %.3f" % DNN.test_accuracy)

def NN_tf_param():
    epochs = 2
    batch_size = 100
    n_neurons_layer1 = 10
    n_neurons_layer2 = 5
    n_categories = 2
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)

    DNN_tf = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            DNN = NNT(X_train_scaled, y_train_onehot, X_test_scaled, y_test_onehot,
                                          n_neurons_layer1=n_neurons_layer1,
                                          n_neurons_layer2=n_neurons_layer2,
                                          n_categories=n_categories, epochs=epochs, batch_size=batch_size, eta=eta,
                                          lmbd=lmbd)
            DNN.fit()

            DNN_tf[i][j] = DNN
            print(DNN)
            print("Learning rate = ", eta)
            print("Lambda = ", lmbd)