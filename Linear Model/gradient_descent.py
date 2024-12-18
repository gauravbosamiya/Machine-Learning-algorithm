import numpy as np 
import random

# BatchGradientDecent
class BatchGradientDecent:
    def __init__(self,learning_rate=0.01,epochs=100):
        self.coef_=None
        self.interncept_=None
        self.lr = learning_rate
        self.epochs=epochs
        
    def fit(self,X_train,y_train):
        self.interncept_=0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            y_hat = np.dot(X_train,self.coef_) + self.interncept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.interncept_ = self.interncept_ - (self.lr * intercept_der)
            
            coef_der = -2 * np.dot((y_train-y_hat),X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
        
            
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.interncept_
    
    
# StochasticGradientDecent
class StochasticGradientDecent :
    def __init__(self,learning_rate=0.01,epochs=100):
        self.coef_=None
        self.interncept_=None
        self.lr = learning_rate
        self.epochs=epochs
        
    def fit(self,X_train,y_train):
        self.interncept_=0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.interncept_
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.interncept_ = self.interncept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
            
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.interncept_
    
    
    
# MiniBatchGradientDecent
class MiniBatchGradientDecent:
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(int(X_train.shape[0]/self.batch_size)):
                
                idx = random.sample(range(X_train.shape[0]),self.batch_size)
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
                
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_