from random import random, seed
import numpy as np 
from NeuralNetwork import NeuralNetwork as NN
from inspect import signature
import time
from sklearn.metrics import mean_squared_error, r2_score
from Misc import CreateDesignMatrix_X
import matplotlib.pyplot as plt
import math as m


seed = 3500

def hypertuning_franke(z, x, y, iterations, cols, etamin, etamax, lmbdmin, lmbdmax, batch_sizemin, batch_sizemax, hiddenmin,hiddenmax, polymin, polymax, plot_MSE = False):

    start_time = time.time()
    if (cols < iterations):
        cols = iterations
        print("cols must be larger than 'iterations. Cols is set equal to iterations")
    sig = signature(hypertuning_franke)
    rows = int(len(sig.parameters)/2)-2
    hyper = np.zeros((rows, cols))
    MSE_array = np.zeros(iterations)
    hyper[0] =  np.logspace(etamin, etamax, cols)
    hyper[1] = np.logspace(lmbdmin, lmbdmax, cols)
    hyper[2] = np.round(np.linspace(batch_sizemin, batch_sizemax, cols, dtype='int'))
    hyper[3] = np.round(np.linspace(hiddenmin, hiddenmax, cols))
    hyper[4] = np.random.randint(polymin, polymax, size=cols, dtype='int')
    hyper[5] = np.zeros((cols))
    

    
    for i in range(rows-1):
        np.random.shuffle(hyper[i])

    n_categories = 1
    

    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]
        X, X_train, X_test, X_val, z_train, z_test, z_val, indicies = CreateDesignMatrix_X(z, x,y, int(hyper[4][it]))
        epochs = 1000

        np.random.seed(seed)
        dnn = NN(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='MSE', activation='sigmoid', activation_out='ELU')
        dnn.train_and_validate(X_val, z_val, MSE_store = plot_MSE, validate=False)

        z_pred = dnn.predict_probabilities(X_val)
        MSE_array[it] = mean_squared_error(z_val,z_pred)
        hyper[5][it] = dnn.epoch +1

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

        if (it%m.ceil((iterations/60))==0):
            print('Iteration: ', it)
            t = round((time.time() - start_time))
            if (t >= 60) and (it > 0):
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
                print("Estimated minutes left: ", int((t/it)*(iterations-it)/60))
            else:
                print("--- %s sec ---" %int(t))
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

"""   
 Overfitting parameters noise = 0.01 for 400 points in Franke function 
[6.15848211e-02] eta
 [5.45559478e-10] lmbd
 [7.90000000e+01] batch size
 [5.90000000e+01] hidden neurons
 [4.00000000e+00] polynomial
 [1.00000000e+03]] epochs

Seed = 3500 
noise = 0.1 
 [[2.06913808e-03]
 [2.63665090e-06]
 [5.30000000e+01]
 [2.60000000e+01]
 [8.00000000e+00]
 [1.00000000e+03]]

overfitting franke 2 noise = 0.1
 [[6.15848211e-02]
 [1.43844989e-11]
 [6.80000000e+01]
 [1.00000000e+00]
 [7.00000000e+00]
 [1.00000000e+03]]

nr 3 0.1
 [[6.95192796e-05]
 [1.27427499e-12]
 [4.20000000e+01]
 [4.30000000e+01]
 [6.00000000e+00]
 [1.00000000e+03]]
"""
def hypertuning_CreditCard(iterations, cols, etamax, etamin, lmbdmax, lmbdmin, batch_sizemax, batch_sizemin, hiddenmax,hiddenmin):

    start_time = time.time()
    if (cols < iterations):
        cols = iterations
        print("cols must be larger than 'iterations. Cols is set equal to iterations")
    sig = signature(hypertuning_CreditCard)
    rows = int(len(sig.parameters)/2)
    hyper = np.zeros((rows, cols))
    AUC_array = np.zeros(iterations)
    hyper[0] =  np.logspace(etamin, etamax, cols)
    hyper[1] = np.logspace(lmbdmin, lmbdmax, cols)
    hyper[2] = np.round(np.linspace(batch_sizemin, batch_sizemax, cols, dtype='int'))
    hyper[3] = np.round(np.linspace(hiddenmin, hiddenmax, cols))
    hyper[4] = np.zeros((cols))

    for i in range(rows-1):
        np.random.shuffle(hyper[i])

    n_categories = 1


    for it in range(iterations):
        hyper_choice = hyper[:,it]
        eta = hyper_choice[0]
        lmbd = hyper_choice[1]
        batch_size = hyper_choice[2]
        n_hidden_neurons = hyper_choice[3]
        epochs = 20

        
        dnn = NN(X_train_scaled, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories,
                 cost_grad='crossentropy', activation='sigmoid', activation_out='sigmoid')
        
        dnn.train_and_validate()
        y_pred = dnn.predict_probabilities(X_test_scaled)

        AUC_array[it] = roc_auc_score(y_test, y_pred)
        
        hyper[4][it] = dnn.epoch +1
        if (it%m.ceil((iterations/40))==0):
            print('Iteration: ', it)
            t = round((time.time() - start_time))
            if t >= 60:
                sec = t % 60
                print("--- %s min," % int(t/60),"%s sec ---" % sec)
                print("Estimated minutes left: ", int(((iterations/it)*t -it*t)/60))
            else:
                print("--- %s sec ---" %int(t))
    AUC_best_index = np.argmax(AUC_array)
    AUC_best = np.max(AUC_array)
    print("AUC array: ", AUC_array)
    print("best index: ",AUC_best_index)
    print("best AUC: ", AUC_best)
    final_hyper = hyper[:,AUC_best_index]

    print("parameters: ", final_hyper)
    eta_best = final_hyper[0]
    lmbd_best = final_hyper[1]
    batch_size_best = final_hyper[2]
    n_hidden_neurons_best = final_hyper[3]
    epochs_best = final_hyper[4]
    return hyper[:,AUC_best_index]