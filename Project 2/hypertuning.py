from random import random, seed
import numpy as np 
from NeuralNetwork import NeuralNetwork as NN
from inspect import signature
import time
from sklearn.metrics import mean_squared_error, r2_score
from Misc import CreateDesignMatrix_X
import matplotlib.pyplot as plt
import math as m
from sklearn.metrics import roc_auc_score, roc_curve, auc

"""This script include the hypertuning functions for Franke and the Credit Card Data.
The parameters that are tuned are eta, lambda, batch size, hidden neurons and in the case of Franke: degree of polynomial.
For the Franke-case MSE was used to find the best tuned parameters, so the combination of these that gave the lowest MSE.
For the Credit Card data-case the AUC was used to find the combination of best parameters, ergo the highest roc_auc_score."""

seed = 3000

# cols is number of columns, which corresponds to number of different parameter-configurations. This must be larger than number of iterations.
# plot_MSE let's you see the MSE plot for every epoch. The function validate_and_early_stopping can be turned off by validate = False. Meaning that it will run to full epoch everytime.
# In that case MSE comparison between train and validation set is still possible(plot_MSE = True and validate = False). Useful to illustrate overfitting.
def hypertuning_franke(z, x, y, iterations, cols, etamin, etamax, lmbdmin, lmbdmax, batch_sizemin, batch_sizemax, hiddenmin,hiddenmax, polymin, polymax, epochs = 1000, plot_MSE = False, validate = True):

    start_time = time.time()
    if (cols < iterations):
        cols = iterations
        print("cols must be larger than 'iterations. Cols is set equal to iterations")
    rows = 6
    hyper = np.zeros((rows, cols))
    MSE_array = np.zeros(iterations)
    #Making the matrix of parameters 
    hyper[0] =  np.logspace(etamin, etamax, cols)
    hyper[1] = np.logspace(lmbdmin, lmbdmax, cols)
    hyper[2] = np.round(np.linspace(batch_sizemin, batch_sizemax, cols, dtype='int'))
    hyper[3] = np.round(np.linspace(hiddenmin, hiddenmax, cols))
    hyper[4] = np.random.randint(polymin, polymax, size=cols, dtype='int')
    hyper[5] = np.zeros((cols))
    

    
    for i in range(rows-1):
        np.random.shuffle(hyper[i])

    n_categories = 1
    
    #iterating over all parameters 
    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]
        X, X_train, X_test, X_val, z_train, z_test, z_val, indicies = CreateDesignMatrix_X(z, x,y, int(hyper[4][it]))

        np.random.seed(seed)
        dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='MSE', activation='sigmoid', activation_out='ELU')
        dnn.train_and_validate(X_val, z_val, MSE_store = plot_MSE, validate=validate)

        z_pred = dnn.predict_probabilities(X_val)
        MSE_array[it] = mean_squared_error(z_val,z_pred)
        hyper[5][it] = dnn.epoch +1

        #Optional: If one wishes to see how the parameter combination is doing, pass plot_MSE = True:
        if(plot_MSE):
            print("parameters: eta, lmbd, batch, hidden, poly, epochs \n", hyper[:,it:it+1])
            MSE_val, MSE_train = dnn.MSE_epoch()
            epo = np.arange(len(MSE_val))
            plt.plot(epo, MSE_val, label='MSE val')
            plt.plot(epo, MSE_train, label='MSE train')
            plt.xlabel("Number of epochs")
            plt.ylabel("MSE")
            plt.title("MSE vs epochs")
            plt.legend()
            plt.show()

        #Estimating the time the iteration takes
        if (it%m.ceil((iterations/60))==0):
            print('Iteration: ', it)
            t = round((time.time() - start_time))
            if (t >= 60) and (it > 0):
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
                print("Estimated minutes left: ", int((t/it)*(iterations-it)/60))
            else:
                print("--- %s sec ---" %int(t))

    # Finding the best parameters:
    MSE_best_index = np.argmin(MSE_array)
    MSE_best = np.min(MSE_array)
    print("MSE array: ", MSE_array)
    print("best index: ",MSE_best_index)
    print("best MSE: ", MSE_best)
    final_hyper = hyper[:,MSE_best_index]

    print("parameters: eta, lmbd, batch, hidden, poly, epochs ", final_hyper)
    eta_best = final_hyper[0]
    lmbd_best = final_hyper[1]
    batch_size_best = final_hyper[2]
    n_hidden_neurons_best = final_hyper[3]
    poly_best = final_hyper[4]
    epochs_best = final_hyper[5]
    return hyper[:,MSE_best_index]

# cols is number of columns, which corresponds to number of different parameters configurations. This must be larger than number of iterations.
def hypertuning_CreditCard(X_train, y_train, X_val, y_val, iterations, cols, etamin, etamax, lmbdmin, lmbdmax, batch_sizemin, batch_sizemax, hiddenmin,hiddenmax, epochs = 5):

    start_time = time.time()
    if (cols < iterations):
        cols = iterations
        print("cols must be larger than 'iterations. Cols is set equal to iterations")
    rows = 5
    hyper = np.zeros((rows, cols))
    AUC_array = np.zeros(iterations)
    #Making the matrix of parameters 
    hyper[0] =  np.logspace(etamin, etamax, cols)
    hyper[1] = np.logspace(lmbdmin, lmbdmax, cols)
    hyper[2] = np.round(np.linspace(batch_sizemin, batch_sizemax, cols, dtype='int'))
    hyper[3] = np.round(np.linspace(hiddenmin, hiddenmax, cols))
    hyper[4] = np.zeros((cols))

    for i in range(rows-1):
        np.random.shuffle(hyper[i])

    n_categories = 1

    #iterating over all parameters 
    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]

        
        dnn = NN(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='crossentropy', activation='sigmoid', activation_out='sigmoid')
        
        dnn.train_and_validate()
        y_pred = dnn.predict_probabilities(X_val)

        AUC_array[it] = roc_auc_score(y_val, y_pred)
        
        hyper[4][it] = dnn.epoch +1

        #Estimating the time the iteration takes
        if (it%m.ceil((iterations/40))==0):
            print('Iteration: ', it)
            t = round((time.time() - start_time))
            if (t >= 60) and (it > 0):
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
                print("Estimated minutes left: ", int((t/it)*(iterations-it)/60))
            else:
                print("--- %s sec ---" %int(t))

    # Finding the best parameters:            
    AUC_best_index = np.argmax(AUC_array)
    AUC_best = np.max(AUC_array)
    print("AUC array: ", AUC_array)
    print("best index: ",AUC_best_index)
    print("best AUC: ", AUC_best)
    final_hyper = hyper[:,AUC_best_index]

    print("parameters: eta, lmbd, batch, hidden, epochs ", final_hyper)
    eta_best = final_hyper[0]
    lmbd_best = final_hyper[1]
    batch_size_best = final_hyper[2]
    n_hidden_neurons_best = final_hyper[3]
    epochs_best = final_hyper[4]
    return hyper[:,AUC_best_index]