
import numpy as np
from scipy import optimize
import pandas as pd
from matplotlib import pyplot

class lr:
    # Data cleaning and finding the mean of the column titled "MaxTemp"
    def data_clean(self,data):
        # 'data' is a dataframe imported from '.csv' file using 'pandas'
        data = data.replace({'RainTomorrow':{'yes':1, 'no':0}}) #a
        data = data.select_dtypes(exclude=['object']) #b
        data = data.fillna(data.mean())
        m, n = data.size
        # Perform data cleaning steps sequentially as mentioned in assignment
        
        
        
        X =   data[:, 0:n-1]          # X (Feature matrix) - should be numpy array
        for i in range(n-1):
            X[:, i] = (X[:, i] - min(X[:, i]))/(max(X[:, i]) - min(X[:, i]))
        y =    data[:, n]         # y (prediction vector) - should be numpy arrays
        norm = X['MaxTemp']
        mean =  np.mean(norm)        # Mean of a the normalized "MaxTemp" column rounded off to 3 decimal places
    
        return X, y, mean

class costing:
    # define the function needed to evaluate cost function
    # Input 'z' could be a scalar or a 1D vector (don't change it, it's correct)
    def sigmoid(self,z):
        z = np.array(z)
        g = np.zeros(z.shape)
        g = 1/(1+np.exp(-z))     
        
        return g
    
    # Regularized cost function definition
    def costFunctionReg(self,w,X,y,lambda_):
        
        m = X.shape[0]
        z = np.dot((np.transpose(w)), X)
        hyp = sigmoid(z)
        J1 = -1*np.matmul(y, log(hyp)) - np.matmul((1 - y),log(1 - hyp))
        J2 = w@w
        J2 = J2*0.5*lambda_
        J = (J1+J2)/m            # Cost 'J' should be a scalar
        grad = np.matmul((hyp-y), x)/m + lambda_*w/m         # Gradient 'grad' should be a vector
        grad[0] = np.matmul((hyp - y), x) / m
        return J, grad
    
    # Prediction based on trained model
    # Use sigmoid function to calculate probability rounded off to either 0 or 1
    def predict(self,w,X):
        
        
        
        z = np.dot((np.transpose(w)), X)
        g = sigmoid(z)
        m = g.size[0]             # 'p' should be a vector of size equal to that of vector 'y'
        p = np.zeros(m)
        for i in range(m):
            p[i] = round(g[i])
        
        return p
    
    # Optimization defintion
    def minCostFun(self, w_ini, X_train, y_train, iters):
        # iters - Maximum no. of iterations; X_train - Numpy array
        lambda_ = 0.1      # Regularization parameter
        m = X.size[0]
        options = {'maxiter': 400}
        X_train =    np.concatenate([np.ones((m, 1)), X_train], axis=1)   # Add '1' for bias term (done)
        res = optimize.minimize(costFunctionreg, w_ini, (X, y), jac = True, method = 'TNC', options=options)
        cost = res.fun
        
        
        w_opt =  res.x       # Optimized weights rounded off to 3 decimal places
        p = predict(w_ini, X_train)
        m = p.size[0]
        corpred=0
        for i in range(m):
            if p[i]==y_train[i]:
                corpred+=1
                
        acrcy =  100*corpred/ m     # Training set accuracy (in %) rounded off to 3 decimal places
        
        return w_opt, acrcy
    
    # Calculate testing accuracy
    def TestingAccu(self, w_opt, X_test, y_test):
        w_opt =  w_opt       # Optimum weights calculated using training data
        m = y_test.shape[0]
        X_test = np.concatenate([np.ones((m, 1)), X_test], axis=1)       # Add '1' for bias term
        
        p = predict(w_ini, X_train)
        m = p.size[0]
        corpred=0
        for i in range(m):
            if p[i]==y_test[i]:
                corpred+=1
        
        acrcy_test =  100*corpred/m  # Testing set accuracy (in %) rounded off to 3 decimal places
        
        return acrcy_test