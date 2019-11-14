import numpy as np
from numpy.linalg import pinv


def sigmoid(X, beta):

    t = X @ beta
    siggy = np.array(1 / (1 + np.exp(-t)),dtype=np.float128)
    return np.nan_to_num(siggy)


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
        

    def gradient_descent(self):
        self.gradient = -(self.X.T.dot(self.y - sigmoid(self.X, self.beta)))
        self.beta -= self.eta * self.gradient
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
        t0, t1 = 1, 120
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
            #self.eta = self.learning_schedule((self.epoch*m+i)*1)
            self.beta -=  (self.eta*gradient)
            self.cost = np.append(self.cost, self.cost_function())

        return self.beta

    def stochastic_gradient_descent(self):
        m = len(self.y_all)
        #random.seed(seed)
        Xshuffled, yshuffled = self.shuffle()
        #print(self.eta)
        self.cost = np.array([])
        for i in range(m):

            self.X = Xshuffled[i:i+1,:]
            self.y = yshuffled[i:i+1,:]
            gradient = -(self.X.T @ (self.y - sigmoid(self.X, self.beta)))
            #self.eta = self.learning_schedule((self.epoch*m+i)*1)
            self.beta -=  self.eta*gradient
            #print(self.beta)
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