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



df['EDUCATION']=np.where(df['EDUCATION'] == 5, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 6, 4, df['EDUCATION'])
df['EDUCATION']=np.where(df['EDUCATION'] == 0, 4, df['EDUCATION'])

df['MARRIAGE']=np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
df['MARRIAGE'].unique()


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

#X_leave, X_use, y_leave, y_use = train_test_split(X, y, test_size = 0.1, stratify=y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size= 0.5, stratify = y_test, random_state=seed)


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
    fig, ax = plt.subplots(figsize=(20, 15))
    corr = df.corr()
    ax = sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True, fmt='.2f', annot_kws={"size": 7})
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'

    );
    ax.set_ylim(df.shape[1], 0)
    ax.set_title("Correlation Matrix")
    plt.show()

#Correlation(df)


############ Logistic Regression with Gradient Descent ##################

eta = 0.0001 #Learning rate

a = np.random.randn(X_train.shape[1],1)
beta_init = np.ones((X_train.shape[1],1))




def sigmoid(X, beta):

    t = X @ beta
    siggy = np.array(1 / (1 + np.exp(-t)),dtype=np.float128)
    return np.nan_to_num(siggy)


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

#Gradient Descent: First we runned with 10 eta values for 500 iterations and got best auc = 0.7019676, and when running for 100 etas vaules of auc we got 0.703562 with eta 3.59381366e-05 
#Newton Raphson's:  
#Stochastic Gradient Descent: Run 20 different etas for 2 iterations and got best auc = 0.7003253147722386, for eta =0.002069138081114788
#Mini_batch_SGD: Run for 50 different etas for 10 iterations and got best auc = 0.7032523897887962, eta =0.0005179474679231208
#  
    def Best_Parameters(self, method, etamin, etamax, step, y_test, X_test):
        method = getattr(self,method)
        eta_vals = np.logspace(etamin, etamax, step)
        print(eta_vals)
        auc_array = np.zeros((2, step))
        for i, eta in enumerate(eta_vals):
            np.random.seed(seed)
            #self.beta = self.beta_init
            self.X = self.X_all
            self.y = self.y_all
            self.eta = eta
            #print(self.eta)
            print(i)
            final_betas,_ = self.train(method)
            prob = sigmoid(X_test, final_betas)
            auc_array[0][i] = roc_auc_score(y_test, prob)

            auc_array[1][i] = eta
        print(auc_array)
        max_auc = np.max(auc_array[0])
        best_eta =  auc_array[1][np.argmax(auc_array[0])]

        return max_auc, best_eta
          


def logistic_regression_SKL(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=seed, solver='lbfgs',multi_class = 'ovr').fit(X_train, y_train.ravel())
    prob_LogReg_Skl = clf.predict_proba(X_test)[:,1:2]
    print(prob_LogReg_Skl)
    false_pos_LogReg_Skl, true_pos_LogReg_Skl = roc_curve(y_test, prob_LogReg_Skl)[0:2]
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

    dnn = MLPClassifier(hidden_layer_sizes=(25),activation='tanh', solver='sgd', alpha=lmbd, batch_size=batch_size, 
        learning_rate_init=eta, 
        max_iter=epochs,random_state=seed)
    
    dnn.fit(X_train_scaled, y_train)
    y_predict = dnn.predict_proba(X_test_scaled)
    
    prob_y = np.zeros((len(y_predict)))
   
    for i in range(len(y_predict)):
        prob_y[i] = np.around(y_predict[i][1])
     
       
    yes = prob_y.sum()
    no = len(prob_y)-yes

    #precentage 
    yes_perc = round(yes/len(prob_y)*100, 1)
    no_perc = round(no/len(prob_y)*100, 1)    
    print( "yes: ", yes_perc)
    print("no: ", no_perc)
    x = np.linspace(0, 1, len(y_predict))
    m = 100/yes_perc
    
    best_line = np.zeros((len(x)))
    for bleh in range(len(x)):
        best_line[bleh] = m*x[bleh]
        if (x[bleh] > yes_perc/100):
            best_line[bleh] = 1


    x_ , y_ = skplt.helpers.cumulative_gain_curve(y_test, y_predict[:,1])
  
    
    Score = (np.trapz(y_, x=x_) - 0.5)/(np.trapz(best_line, dx=(1/len(y_predict)))-0.5)
    print(Score)  #Area ratio 0.4791027525012573

#SciKitLearn_Classification()

def Plots(ROC_plot, Lift_plot_test_NN, Lift_plot_train_NN, Lift_plot_MB, GD_plot, MB_GD_plot, Stoch_GD_plot, Newton_plot, Scatter_GD_plot):
    seed = 3000
    if (ROC_plot==1):
        
        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w = Weight(X_train_scaled,y_train,beta_init,3.59381366e-05, 20)  
        final_betas_grad,_ = w.train(w.gradient_descent)
        prob_grad, y_pred_grad = classification(X_test_scaled, final_betas_grad, y_test)[0:2]
        false_pos_grad, true_pos_grad = roc_curve(y_test, prob_grad)[0:2]
        print("Area under curve gradient: ", auc(false_pos_grad, true_pos_grad))
       
        #
        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w1 = Weight(X_train_scaled,y_train,beta_init,0.0005179474679231208, 20)
        final_betas_MB,_ = w1.train(w1.mini_batch_gradient_descent)
        prob_MB, y_pred_MB = classification(X_test_scaled,final_betas_MB, y_test)[0:2]
        false_pos_MB, true_pos_MB = roc_curve(y_test, prob_MB)[0:2]
        print("Area under curve MB: ", auc(false_pos_MB, true_pos_MB))

        np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w2 = Weight(X_train_scaled,y_train,beta_init,0.002069138081114788, 20)
        final_betas_ST,_ = w2.train(w2.stochastic_gradient_descent)
        prob_ST, y_pred_ST = classification(X_test_scaled,final_betas_ST, y_test)[0:2]
        false_pos_ST, true_pos_ST = roc_curve(y_test, prob_ST)[0:2]
        print("Area under curve ST: ", auc(false_pos_ST, true_pos_ST))

        # IMPORTANT INFORMATION DONT TAKE IT AWAY!! #####
        #First roc_curve produced was with optimal found eta for all the methods but only for 3 iterations to show how much more efficent stochastic and mini-batch
        #is compared to gradient and newton.
        #In second run roc_curve produced is with the optimal iterations (300) for gradient as well, newton dropped because it is so slow. 
        #Area under curve gradient:  0.7005858407946856 iterations 300
        #Area under curve MB:  0.7009378481391317 iterations 20
        #Area under curve ST:  0.7038241794231722 iterations 30
        #Area under curve NN:  0.7710878832109259 iterations 20, eta = 0.1, lmbd = 0.01, batch_size=25, n_hidden = 41
        # AUC for iteration 20 for all:
        #Area under curve gradient:  0.6140996993109424
        #Area under curve MB:  0.7009378481391317
        #Area under curve ST:  0.7038241794231722
        #Area under curve NN:  0.7710878832109259


        #np.random.seed(seed) 
        # final_betas_ST_Skl,_ = w.train(w.stochastic_gradient_descent_Skl)
        # prob_ST_Skl, y_pred_ST_Skl = classification(X_test_scaled,final_betas_ST_Skl, y_test)[0:2]
        # false_pos_ST_Skl, true_pos_ST_Skl = roc_curve(y_test, prob_ST_Skl)[0:2]
        # print("Area under curve ST_skl: ", auc(false_pos_ST_Skl, true_pos_ST_Skl))

        """np.random.seed(seed)
        beta_init = np.random.randn(X_train.shape[1],1)
        w3 = Weight(X_train_scaled,y_train,beta_init,0.001, 20)
        final_betas_Newton,_ = w3.train(w3.newtons_method)
        prob_Newton, y_pred_Newton = classification(X_test_scaled,final_betas_Newton, y_test)[0:2]
        false_pos_Newton, true_pos_Newton = roc_curve(y_test, prob_Newton)[0:2]
        print("Area under curve Newton: ", auc(false_pos_Newton, true_pos_Newton))"""
        
        np.random.seed(seed)
        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41
        n_categories = 1

        dnn = NN(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'crossentropy', activation = 'sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()
    
        y_predict = dnn.predict_probabilities(X_test_scaled)
        
        false_pos_NN, true_pos_NN = roc_curve(y_test, y_predict)[0:2]
        print("Area under curve NN: ", auc(false_pos_NN, true_pos_NN))

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(false_pos_grad, true_pos_grad,label="Gradient")
        plt.plot(false_pos_ST, true_pos_ST, label="Stoch")
        #plt.plot(false_pos_ST_Skl, true_pos_ST_Skl, label="Stoch_Skl")
        plt.plot(false_pos_MB, true_pos_MB, label="Mini")
        #plt.plot(false_pos_Newton, true_pos_Newton, label="Newton")
        plt.plot(false_pos_NN, true_pos_NN, label='NeuralNetwork')
        plt.legend()
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.title("ROC curve")
        plt.show()



    if (Lift_plot_test_NN == 1):

        np.random.seed(seed)
        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41
        n_categories = 1

        dnn = NN(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'crossentropy', activation = 'sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()
        
        y_predict = dnn.predict_probabilities(X_test_scaled)

        y_pred = np.concatenate((1-y_predict,y_predict),axis=1)
        prob_y = np.zeros((len(y_predict)))
        print(y_predict.shape)
        for i in range(len(y_predict)):
            prob_y[i] = np.around(y_predict[i])
            
        yes = prob_y.sum()
        no = len(prob_y)-yes

        #precentage 
        yes_perc = round(yes/len(prob_y)*100, 1)
        no_perc = round(no/len(prob_y)*100, 1)    
        print( "yes: ", yes_perc)
        print("no: ", no_perc)
        x = np.linspace(0, 1, len(y_predict))
        m = 100/yes_perc
        
        best_line = np.zeros((len(x)))
        for bleh in range(len(x)):
            best_line[bleh] = m*x[bleh]
            if (x[bleh] > yes_perc/100):
                best_line[bleh] = 1


        x_ , y_ = skplt.helpers.cumulative_gain_curve(y_test, y_pred[:,1])
      
        
        Score = (np.trapz(y_, x=x_) - 0.5)/(np.trapz(best_line, dx=(1/len(y_predict)))-0.5)
        print(Score) # The score  Area ratio = 0.49129354889528054 Neural Network test against predicted
        perc = np.linspace(0,100,len(x))
        plt.plot(np.linspace(0,100,len(x_)), y_)
        plt.plot(perc, best_line)
        plt.plot(perc, np.linspace(0,1, len(x)), "k--")
        
        plt.xlabel("Precentage of people")
        plt.ylabel("Default payment")
        plt.title("Lift Chart for Test")
        plt.show()

    if (Lift_plot_train_NN == 1):

        np.random.seed(seed)
        epochs = 20
        batch_size = 25
        eta = 0.1
        lmbd = 0.01
        n_hidden_neurons = 41
        n_categories = 1

        dnn = NN(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                    cost_grad = 'crossentropy', activation = 'sigmoid', activation_out='sigmoid')
        dnn.train_and_validate()
        
        y_predict = dnn.predict_probabilities(X_train_scaled)
        y_pred = np.concatenate((1-y_predict,y_predict),axis=1)
        prob_y = np.zeros((len(y_predict)))
        
        for i in range(len(y_predict)):
            prob_y[i] = np.around(y_predict[i])
            
        yes = prob_y.sum()
        no = len(prob_y)-yes

         
        yes_perc = round(yes/len(prob_y)*100, 1)
        no_perc = round(no/len(prob_y)*100, 1)    
        print( "yes: ", yes_perc)
        print("no: ", no_perc)
        x = np.linspace(0, 1, len(y_predict))
        m = 100/yes_perc
        
        best_line = np.zeros((len(x)))
        for bleh in range(len(x)):
            best_line[bleh] = m*x[bleh]
            if (x[bleh] > yes_perc/100):
                best_line[bleh] = 1


        x_ , y_ = skplt.helpers.cumulative_gain_curve(y_train, y_pred[:,1])
      
        
        Score = (np.trapz(y_, x=x_) - 0.5)/(np.trapz(best_line, dx=(1/len(y_predict)))-0.5)
        print(Score) # The score  Area ratio = 0.49167470576088995 Neural Network train against predicted
        perc = np.linspace(0,100,len(x))
        plt.plot(np.linspace(0,100,len(x_)), y_)
        plt.plot(perc, best_line)
        plt.plot(perc, np.linspace(0,1, len(x)), "k--")
        
        plt.xlabel("Precentage of people")
        plt.ylabel("Default payment")
        plt.title("Lift Chart for Train")
        plt.show()    

    if (Lift_plot_MB == 1):

        np.random.seed(seed)

        beta_init = np.random.randn(X_train.shape[1],1)
        w1 = Weight(X_train_scaled,y_train,beta_init,3.59381366e-05, 350)

        final_betas_MB,_ = w1.train(w1.gradient_descent)
        prob_MB, y_pred_MB = classification(X_test_scaled,final_betas_MB, y_test)[0:2]
        
        y_pred = np.concatenate((1-prob_MB,prob_MB),axis=1)
        
        prob_y = prob_y = np.zeros((len(prob_MB)))
        for i in range(len(prob_MB)):
            prob_y[i] = np.around(prob_MB[i])
            
        yes = prob_y.sum()
        no = len(prob_MB)-yes

         
        yes_perc = round(yes/len(y_pred_MB)*100, 1)
        no_perc = round(no/len(y_pred_MB)*100, 1)    
        print( "yes: ", yes_perc)
        print("no: ", no_perc)
        x = np.linspace(0, 1, len(y_pred_MB))
        m = 100/yes_perc
        
        best_line = np.zeros((len(x)))
        for bleh in range(len(x)):
            best_line[bleh] = m*x[bleh]
            if (x[bleh] > yes_perc/100):
                best_line[bleh] = 1


        x_ , y_ = skplt.helpers.cumulative_gain_curve(y_train, y_pred[:,1])
      
        
        Score = (np.trapz(y_, x=x_) - 0.5)/(np.trapz(best_line, dx=(1/len(y_pred_MB)))-0.5)
        print(Score) # The score  Area ratio = 0.49167470576088995 Neural Network train against predicted
        perc = np.linspace(0,100,len(x))
        plt.plot(np.linspace(0,100,len(x_)), y_)
        plt.plot(perc, best_line)
        plt.plot(perc, np.linspace(0,1, len(x)), "k--")
        
        plt.xlabel("Precentage of people")
        plt.ylabel("Default payment")
        plt.title("Lift Chart for Train")
        plt.show()
            
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


Plots(0, 0, 0, 1, 0, 0, 0, 0, 0)


#print(hypertuning_CreditCard(10, 10, 1, -4, 1, -4, 100, 1, 100, 1))

