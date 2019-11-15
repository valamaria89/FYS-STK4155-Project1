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
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from functools import partial
import tensorflow as tf
import seaborn as sns
from NeuralNetwork import NeuralNetwork as NN
import time
from inspect import signature
import math as m
from hypertuning import hypertuning_CreditCard
from Weight import Weight as Weight

import random

seed = 3000
np.random.seed(seed)

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

df['EDUCATION'] = np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION'] = np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION'] = np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
df['MARRIAGE'].unique()

X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

yes = df.defaultPaymentNextMonth.sum()
no = len(df) - yes

# precentage of defaults non-defaults in the original dataset
yes_perc = round(yes / len(df) * 100, 1)
no_perc = round(no / len(df) * 100, 1)

print("Default: ", yes_perc)
print("Non-Default: ", no_perc)

"""Scales the data set"""
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=seed)


"""Creates the correlation matrix for a pandas data set"""
def Correlation(df):
    del df['defaultPaymentNextMonth']
    sns.set()
    fig, ax = plt.subplots(figsize=(20, 15))
    corr = df.corr()
    ax = sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
                     square=True, fmt='.2f', annot_kws={"size": 7})
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'

    );
    ax.set_ylim(df.shape[1], 0)
    ax.set_title("Correlation Matrix")
    plt.show()


############ Logistic Regression with Gradient Descent and Neural Networks ##################


def sigmoid(X, beta, beta_intercept=0):

    t = X @ beta + beta_intercept
    siggy = np.array(1 / (1 + np.exp(-t)), dtype=np.float128)
    return np.nan_to_num(siggy)


"""Let's you put in a threshold and classify probabilities"""
def classification(X = None, betas = None, y_test=None, y_prob_input = None, threshold = 0.5):
    y_pred = 0
    prob = 0
    tot = 0
    if (betas is not None) and (X is not None) and (y_prob_input is None):
        prob = sigmoid(X, betas)
        y_pred = np.zeros(X.shape[0])
    elif y_prob_input is not None:
        y_pred = np.zeros(y_prob_input.shape[0])
        prob = y_prob_input
    else: print("Insert either finished probability predictions or features and weights")

    for i in range(len(y_pred)):
        if prob[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
        if ((y_test is not None) and (y_pred[i] == y_test[i])):
            tot += 1

    return prob, y_pred, tot / len(y_pred)


"""Checks the best parameters, i.e. grid search, for a given logistic regression method"""
def Best_Parameters(epochs, batch_size, method, etamin, etamax, step, y_val, X_val):
    beta_init = np.random.randn(X_train.shape[1], 1)
    eta_vals = np.logspace(etamin, etamax, step)
    auc_array = np.zeros((2, step))
    for i, eta in enumerate(eta_vals):
        np.random.seed(seed)
        print("Iteration: ",i)
        print("eta: ", eta)
        w = Weight(X_train, y_train, beta_init, eta, epochs, batch_size=batch_size)
        method_ = getattr(w, method)
        final_betas, _ = w.train(method_)
        prob = sigmoid(X_val, final_betas)
        auc_array[0][i] = roc_auc_score(y_val, prob)

        auc_array[1][i] = eta
    max_auc = np.max(auc_array[0])
    best_eta = auc_array[1][np.argmax(auc_array[0])]

    return max_auc, best_eta


def logistic_regression_SKL(X_train, X_test, y_train, y_test):
    eta = 0.0007924828983539169
    clf = SGDClassifier(loss='log', penalty='None', max_iter=20, eta0=eta, learning_rate='constant', fit_intercept=True,
                        random_state=seed)
    clf.fit(X_train, y_train.ravel())
    final_betas_LGSKL = clf.coef_.T
    prob_LGSKL = clf.predict_proba(X_test)[:, 1]
    false_pos_LogReg_Skl, true_pos_LogReg_Skl = roc_curve(y_test, prob_LGSKL)[0:2]
    print("Area under curve LogReg_skl: ", auc(false_pos_LogReg_Skl, true_pos_LogReg_Skl))

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(false_pos_LogReg_Skl, true_pos_LogReg_Skl, label="LogReg")
    plt.legend()
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.title("ROC curve")
    plt.show()
    return


def SciKitLearn_Classification():
    epochs = 20
    batch_size = 25
    eta = 0.1
    lmbd = 0.01
    n_hidden_neurons = 41
    n_categories = 1

    dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons,), activation='logistic', solver='sgd', alpha=0.01,
                        batch_size=batch_size,
                        learning_rate_init=eta,
                        max_iter=epochs, random_state=seed, momentum=0, shuffle=False)

    dnn.fit(X_train, y_train.ravel())
    y_predict = dnn.predict_proba(X_train)

    prob_y = np.zeros((len(y_predict)))

    for i in range(len(y_predict)):
        prob_y[i] = np.around(y_predict[i][1])

    yes = prob_y.sum()
    no = len(prob_y) - yes

    yes_perc = round(yes / len(prob_y) * 100, 1)
    no_perc = round(no / len(prob_y) * 100, 1)
    print("yes: ", yes_perc)
    print("no: ", no_perc)
    x = np.linspace(0, 1, len(y_predict))
    m = 100 / yes_perc

    best_line = np.zeros((len(x)))
    for bleh in range(len(x)):
        best_line[bleh] = m * x[bleh]
        if (x[bleh] > yes_perc / 100):
            best_line[bleh] = 1

    x_, y_ = skplt.helpers.cumulative_gain_curve(y_test, y_predict[:, 1])

    Score = (np.trapz(y_, x=x_) - 0.5) / (np.trapz(best_line, dx=(1 / len(y_predict))) - 0.5)
    print("Area ratio score Scikit NN: ", Score)
    AUC = roc_auc_score(y_test, y_predict[:, 1])
    print("AUC score Scikit NN: ", AUC)


"""Mini batch function set outside of Plots for the purpose of running the GRAD_SGD_MB_plot function through the Plots function."""
def mini_batch_SGD(eta, batch_size, epochs):
    MB_start_time = time.time()
    np.random.seed(seed)
    beta_init = np.random.randn(X_train.shape[1], 1)
    w1 = Weight(X_train, y_train, beta_init, eta, epochs, batch_size=batch_size)
    final_betas_MB, _ = w1.train(w1.mini_batch_gradient_descent)
    prob_MB, y_pred_MB = classification(X_test, final_betas_MB, y_test)[0:2]
    false_pos_MB, true_pos_MB = roc_curve(y_test, prob_MB)[0:2]
    AUC_MB = auc(false_pos_MB, true_pos_MB)
    print("Area under curve MB%s: " %batch_size, AUC_MB)
    MB_time = time.time() - MB_start_time
    return AUC_MB, MB_time, false_pos_MB, true_pos_MB


"""ROC plot puts all the gradient functions and neural network into one graph for comparison"""
"""The cost functions and plot for the various optimisers, and a scatter function: GD_plot = 0, MB_GD_plot = 0, Stoch_GD_plot = 0, Newton_plot = 0, Scatter_GD_plot = 0 """
def Plots(epochs, AUC_time_plot = 0, ROC_plot = 0, Lift_plot_test_NN = 0, Lift_plot_train_NN = 0, GD_plot = 0, MB_GD_plot = 0, Stoch_GD_plot = 0,
          Newton_plot = 0, Scatter_GD_plot = 0):

    if (ROC_plot == 1 or AUC_time_plot == 1):
        GRAD_start_time = time.time()
        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w = Weight(X_train,y_train,beta_init,6.892612104349695e-05, epochs)
        final_betas_grad,cost = w.train(w.gradient_descent)
        prob_grad, y_pred_grad = classification(X_test, final_betas_grad, y_test)[0:2]
        false_pos_grad, true_pos_grad = roc_curve(y_test, prob_grad)[0:2]
        AUC_GRAD = auc(false_pos_grad, true_pos_grad)
        print("Area under curve gradient: ", AUC_GRAD)
        GRAD_time = time.time() - GRAD_start_time

        SGD_start_time = time.time()
        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1], 1)
        w2 = Weight(X_train, y_train, beta_init, 0.0007924828983539169, epochs)
        final_betas_ST, _ = w2.train(w2.stochastic_gradient_descent)
        prob_ST, y_pred_ST = classification(X_test, final_betas_ST, y_test)[0:2]  ### HERE
        false_pos_ST, true_pos_ST = roc_curve(y_test, prob_ST)[0:2]
        AUC_SGD = auc(false_pos_ST, true_pos_ST)
        print("Area under curve ST: ", AUC_SGD)
        SGD_time = time.time() - SGD_start_time

        """np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w3 = Weight(X_train,y_train,beta_init,0.001, 20)
        final_betas_Newton,_ = w3.train(w3.newtons_method)
        prob_Newton, y_pred_Newton = classification(X_train,final_betas_Newton, y_test)[0:2]
        false_pos_Newton, true_pos_Newton = roc_curve(y_test, prob_Newton)[0:2]
        print("Area under curve Newton: ", auc(false_pos_Newton, true_pos_Newton))"""

        AUC_MB5 = 0
        MB5_time = 0
        AUC_MB1000 = 0
        MB1000_time = 0
        AUC_MB6000 = 0
        MB6000_time = 0
        AUC_MB = 0
        false_pos_MB = 0
        true_pos_MB = 0
        if(AUC_time_plot != 0):
            AUC_MB5, MB5_time, _, _ = mini_batch_SGD(0.0038625017292608175, 5, epochs)
            AUC_MB1000, MB1000_time, _, _ = mini_batch_SGD(0.0009501185073181439, 1000, epochs)
            AUC_MB6000, MB6000_time, _ ,_ = mini_batch_SGD(0.0001999908383831537, 6000, epochs)
            return AUC_SGD, AUC_GRAD, AUC_MB5, AUC_MB1000, AUC_MB6000, SGD_time, GRAD_time, MB5_time, MB1000_time, MB6000_time
        else:
            AUC_MB, _,false_pos_MB, true_pos_MB  = mini_batch_SGD(0.0038625017292608175, 32, epochs)

        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1], 1)
        w4 = Weight(X_train, y_train, beta_init, 0.0007924828983539169, epochs)
        final_betas_ST_Skl,_ = w.train(w4.stochastic_gradient_descent_Skl)
        prob_ST_Skl, y_pred_ST_Skl = classification(X_test,final_betas_ST_Skl[0], y_test)[0:2]
        false_pos_ST_Skl, true_pos_ST_Skl = roc_curve(y_test, prob_ST_Skl)[0:2]
        print("Area under curve ST_skl: ", auc(false_pos_ST_Skl, true_pos_ST_Skl))

        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41
        ####################
        # epochs = 20
        # batch_size = 26
        # eta = 3.14230708e+00
        # lmbd = 1.25472709e-02
        # n_hidden_neurons = 66

        np.random.seed(seed)
        n_categories = 1

        dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'crossentropy', activation = 'sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()

        y_predict = dnn.predict_probabilities(X_test)

        false_pos_NN, true_pos_NN = roc_curve(y_test, y_predict)[0:2]
        print("AUC score NN: ", auc(false_pos_NN, true_pos_NN))

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(false_pos_grad, true_pos_grad,label="Gradient")
        plt.plot(false_pos_ST, true_pos_ST, label="Stoch")
        plt.plot(false_pos_ST_Skl, true_pos_ST_Skl, label="Stoch_Skl")
        plt.plot(false_pos_MB, true_pos_MB, label="Mini")
        # plt.plot(false_pos_Newton, true_pos_Newton, label="Newton")
        plt.plot(false_pos_NN, true_pos_NN, label='NeuralNetwork')
        plt.legend()
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.title("ROC curve")
        plt.show()

    """Creates cumulative gain charts/lift plots for Neural network. The two optimal parameters sets from tuning are listed below"""
    if (Lift_plot_test_NN == 1):

        np.random.seed(seed)

        # epochs = 20
        # batch_size = 26
        # eta = 3.14230708e+00
        # lmbd = 1.25472709e-02
        # n_hidden_neurons = 66
        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41

        n_categories = 1

        dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                 n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='crossentropy', activation='sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()

        y_predict_proba = dnn.predict_probabilities(X_test)
        y_predict_proba_tuple = np.concatenate((1 - y_predict_proba, y_predict_proba), axis=1)

        pos_true = y_test.sum()
        pos_true_perc = pos_true / len(y_test)

        x = np.linspace(0, 1, len(y_test))
        m = 1 / pos_true_perc

        best_line = np.zeros((len(x)))
        for i in range(len(x)):
            best_line[i] = m * x[i]
            if (x[i] > pos_true_perc):
                best_line[i] = 1

        x_, y_ = skplt.helpers.cumulative_gain_curve(y_test, y_predict_proba_tuple[:, 1])

        Score = (np.trapz(y_, x=x_) - 0.5) / (np.trapz(best_line, dx=(1 / len(y_predict_proba))) - 0.5)
        print('Area ratio score(test)', Score)  # The score  Area ratio = 0.49129354889528054 Neural Network test against predicted
        perc = np.linspace(0, 100, len(y_test))
        plt.plot(x_*100, y_*100)
        plt.plot(perc, best_line*100)
        plt.plot(perc, perc, "k--")

        plt.xlabel("Percentage of clients")
        plt.ylabel("Cumulative % of defaults")
        plt.title("Cumulative Gain Chart for Test Data")
        plt.show()

        """Let's you insert a threshold and classify"""
        _, y_predict, y_predict_tot = classification(y_prob_input=y_predict_proba, threshold=0.5)
        pos = y_predict.sum()
        neg = len(y_predict) - pos
        pos_perc = (pos / len(y_predict))
        neg_perc = (neg / len(y_predict))
        print("default: ", pos_perc)
        print("Non-default: ", neg_perc)

    if (Lift_plot_train_NN == 1):

        np.random.seed(seed)

        # epochs = 20
        # batch_size = 26
        # eta = 3.14230708e+00
        # lmbd = 1.25472709e-02
        # n_hidden_neurons = 66
        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41
        n_categories = 1

        dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                 n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='crossentropy', activation='sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()

        y_predict_proba = dnn.predict_probabilities(X_train)
        y_predict_proba_tuple = np.concatenate((1 - y_predict_proba, y_predict_proba), axis=1)

        pos_true = y_train.sum()
        pos_true_perc = pos_true / len(y_train)

        x = np.linspace(0, 1, len(y_train))
        m = 1 / pos_true_perc

        best_line = np.zeros((len(x)))
        for i in range(len(x)):
            best_line[i] = m * x[i]
            if (x[i] > pos_true_perc):
                best_line[i] = 1

        x_, y_ = skplt.helpers.cumulative_gain_curve(y_train, y_predict_proba_tuple[:, 1])

        Score = (np.trapz(y_, x=x_) - 0.5) / (np.trapz(best_line, dx=(1 / len(y_predict_proba))) - 0.5)
        print('Area ratio score(train)', Score)
        perc = np.linspace(0, 100, len(y_train))
        plt.plot(x_ * 100, y_ * 100)
        plt.plot(perc, best_line * 100)
        plt.plot(perc, perc, "k--")

        plt.xlabel("Percentage of clients")
        plt.ylabel("Cumulative % of defaults")
        plt.title("Cumulative Gain Chart for Train Data")
        plt.show()

        """Let's you insert a threshold and classify"""
        _, y_predict, y_predict_tot = classification(y_prob_input=y_predict_proba, threshold=0.5)
        pos = y_predict.sum()
        neg = len(y_predict) - pos
        pos_perc = (pos / len(y_predict))
        neg_perc = (neg / len(y_predict))
        print("default: ", pos_perc)
        print("Non-default: ", neg_perc)

    beta_init = np.random.randn(X_train.shape[1], 1)
    w = Weight(X_train, y_train, beta_init, 0.0007924828983539169, epochs)

    if (GD_plot == 1):
        _, cost_all = w.train(w.gradient_descent)
        epoch = np.arange(len(cost_all))

        plt.plot(epoch, cost_all)
        plt.show()

    if (MB_GD_plot == 1):
        _, cost_all = w.train(w.mini_batch_gradient_descent)
        batch = np.arange(len(cost_all))

        plt.plot(batch, cost_all)
        plt.show()

    if (Stoch_GD_plot == 1):
        _, cost_all = w.train(w.stochastic_gradient_descent)
        batch = np.arange(len(cost_all))

        plt.plot(batch, cost_all)
        plt.show()

    if (Newton_plot == 1):
        _, cost_all = w.train(w.newtons_method)
        epochs = np.arange(len(cost_all))

        plt.plot(epochs, cost_all)
        plt.show()

    if (Scatter_GD_plot == 1):
        final_betas, _ = w.train(w.gradient_descent)
        prob_train = classification(X_train, final_betas)[0]
        x_sigmoid = np.dot(X_train, final_betas)
        plt.scatter(x_sigmoid, prob_train)
        plt.show()





"""This function compares with plots the AUC convergence and time used for several different gradient methods. E.g MB1000 means Mini-batch gradient descent with mini bath size of 1000 """
def GRAD_SGD_MB_plot(epochs):
    start_time = time.time()
    AUC_and_time = np.zeros((10,epochs))
    epoch_array = np.arange(epochs)
    for i in range(epochs):
        print("epochs used: ", i)
        AUC_SGD, AUC_GRAD, AUC_MB5, AUC_MB1000, AUC_MB6000, SGD_time, GRAD_time, MB5_time, MB1000_time, MB6000_time = Plots(epochs = i, AUC_time_plot=1)
        AUC_and_time[0][i] = AUC_SGD
        AUC_and_time[1][i] = AUC_GRAD
        AUC_and_time[2][i] = AUC_MB5
        AUC_and_time[3][i] = AUC_MB1000
        AUC_and_time[4][i] = AUC_MB6000
        AUC_and_time[5][i] = SGD_time
        AUC_and_time[6][i] = GRAD_time
        AUC_and_time[7][i] = MB5_time
        AUC_and_time[8][i] = MB1000_time
        AUC_and_time[9][i] = MB6000_time
    print("Total time used: ", time.time() - start_time)
    plt.figure()
    plt.subplot(211)
    plt.plot(epoch_array, AUC_and_time[0], label="AUC SGD", color='tab:orange')
    plt.plot(epoch_array, AUC_and_time[1], label="AUC GRAD", color='tab:blue')
    plt.plot(epoch_array, AUC_and_time[2], label="AUC MB5", color='tab:red')
    plt.plot(epoch_array, AUC_and_time[3], label="AUC MB1000", color='tab:green')
    plt.plot(epoch_array, AUC_and_time[4], label="AUC MB6000", color='tab:purple')
    plt.legend()
    plt.ylabel("AUC-score")
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 1.0))
    plt.subplot(212)
    plt.plot(epoch_array, AUC_and_time[5], label="seconds taken SGD", color='tab:orange', linestyle='--')
    plt.plot(epoch_array, AUC_and_time[6], label="seconds GRAD", color='tab:blue', linestyle='--')
    plt.plot(epoch_array, AUC_and_time[7], label="seconds MB5", color='tab:red', linestyle='--')
    plt.plot(epoch_array, AUC_and_time[8], label="seconds MB1000", color='tab:green', linestyle='--')
    plt.plot(epoch_array, AUC_and_time[9], label="seconds MB6000", color='tab:purple', linestyle='--')
    plt.legend()
    plt.ylabel("Seconds")
    plt.xlabel("Number of epochs iterated over, respectively")
    plt.xticks(np.arange(min(epoch_array), max(epoch_array) + 1, 1.0))
    plt.show()



# GRAD_SGD_MB_plot(21)
# logistic_regression_SKL(X_train, X_test, y_train, y_test)

"""hypertuning_CreditCard(X_train, y_train, X_val, y_val, iterations, cols, etamin, etamax, lmbdmin, lmbdmax, batch_sizemin, batch_sizemax, hiddenmin,hiddenmax):"""

"""Run hypertuning over the parameters """
# hypertuning_CreditCard(X_train, y_train, X_val, y_val, 20, 20, -4, 1, -4, 1, 1, 100, 1, 80, epochs= 3)

"""See the lift plots"""
# Plots(epochs = 20, ROC_plot = 0, Lift_plot_test_NN = 1, Lift_plot_train_NN = 1, GD_plot = 0, MB_GD_plot = 0, Stoch_GD_plot = 0, Newton_plot = 0, Scatter_GD_plot = 0)

"""Check best parameters for minibatch gd"""
# print(Best_Parameters(epochs = 20, batch_size = 32, method = "mini_batch_gradient_descent", etamin = -6, etamax = 0.7, step=100, y_val=y_val, X_val=X_val))

"""Show Roc plot"""
# Plots(epochs = 3, ROC_plot = 1, Lift_plot_test_NN = 0, Lift_plot_train_NN = 0, GD_plot = 0, MB_GD_plot = 0, Stoch_GD_plot = 0, Newton_plot = 0, Scatter_GD_plot = 0)

