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

df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE']=np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
df['MARRIAGE'].unique()

X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


# Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
# y_train_onehot, y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)


class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            cost_grad = 'crossentropy',
            activation = 'sigmoid'):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.X_data = X_data
        self.Y_data = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.cost_grad = getattr(self,cost_grad)
        self.activation = getattr(self,activation)
        self.activation_grad = getattr(self,activation + '_grad')

        self.create_biases_and_weights()

    def sigmoid(self, z):
        siggy = 1 / (1 + np.exp(-z))
        return siggy

    def crossentropy(self, a, y):
        return np.nan_to_num(a-y)/(a*(1-a))

    def sigmoid_grad(self, a):
        return np.nan_to_num(a*(1-a))

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)*1e-3
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)*1e-3
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        # print(np.matmul(self.X_data, self.hidden_weights))
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        self.a_o = self.activation(self.z_o)

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.activation(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        a_o = self.activation(z_o)

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return a_o

    def backpropagation(self):
        error_output = self.activation_grad(self.activation(self.z_o))*self.cost_grad(self.a_o,self.Y_data)
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.activation_grad(self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)

        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):

        data_indices = np.arange(self.n_inputs)
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
                print("I :",i)
                print("J :",j)

                self.feed_forward()
                self.backpropagation()



epochs = 10
batch_size = 10
eta = 0.01
lmbd = 0.01
n_hidden_neurons = 10
n_categories = 2

dnn = NeuralNetwork(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'crossentropy', activation = 'sigmoid')
dnn.train()
test_predict = dnn.predict(X_test_scaled)
test_predict1 = dnn.predict_probabilities(X_test_scaled)[:,1:2]
#
# accuracy score from scikit library
#print("Accuracy score on test set: ", accuracy_score(y_test, test_predict))
#
# def accuracy_score_numpy(Y_test, Y_pred):
#     return np.sum(Y_test == Y_pred) / len(Y_test)

#print(test_predict1)

false_pos, true_pos = roc_curve(y_test, test_predict1)[0:2]
print("Area under curve ST: ", auc(false_pos, true_pos))
plt.plot([0, 1], [0, 1], "k--")
plt.plot(false_pos, true_pos)
plt.xlabel("False Positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC curve gradient descent")
plt.show()