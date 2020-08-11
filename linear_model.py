import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier:
    def __init__(self, verbose = 0, maxEvals = 400):
        self.verbose = verbose
        self.maxEvals = maxEvals
 
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1
         
            self.W[i], f = findMin.findMin(self.funObj, self.W[i], self.maxEvals, X, ytmp)
      

    def funObj(self,w,X,ytmp):
        n, d = X.shape
        f_1 = np.zeros(n)
        g = np.zeros((d,n))
        e = np.zeros(n)
        G = np.zeros(d)
       # calculate the function value
           
        for j in range(d):
            for i in range(n):
                e[i] = np.exp(-ytmp[i]*(w.T@X[i]))
                g[j, i] = (e[i]*(-ytmp[i]*X[i,j]))/(1 + e[i])
        
        for i in range(n):
            f_1[i] = np.log(1 + e[i])
        f = np.sum(f_1)
        G = np.sum(g,axis = 1)
        
        return (f, G)

      #  a = np.exp(-ytmp.T@(X@w))
      #  f = np.sum(np.log(1 + a))
        # calculate the gradient
       
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class softmaxClassifier:
    def __init__(self, verbose = 1, maxEvals = 500):
        self.verbose = verbose
        self.maxEvals = maxEvals
    
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        m = self.n_classes*d
        self.w = np.reshape(self.W, m)
         
        self.w, f = findMin.findMin(self.funObj, self.w, self.maxEvals, X, y)
        
        self.W = np.reshape(self.w,(self.n_classes,d))
        
    def funObj(self, W, X, y):
            n, d = X.shape
            W = np.reshape(W,(self.n_classes,d))
            #I = np.unique(y)
            G = np.zeros((self.n_classes, d))
            k = self.n_classes
            f_1 = np.zeros(k)
            #p_1 = np.zeros(k)
            I = np.unique(y)
            for k in range(self.n_classes):
                I_k = np.where(y == I[k])
                f_1[k] = np.sum(X[I_k]@W[k].T)
                #calculate the probability of  each example being in class c
                p_1 = np.exp(X@W[k].T)/np.sum(np.exp(X@W.T), axis = 1)
                for j in range(d):
                    #compute the gradient
                    G[k,j] = - np.sum(X[I_k, j]) + p_1.T@X[:,j]
            #compute the function       
            F = - np.sum(f_1)+ np.sum(np.log(np.sum(np.exp(X@W.T), axis = 1)))
            G = G.flatten()
            return (F, G)
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)