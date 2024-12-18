import numpy as np

# Simple Linear Regression for 2D Data
class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,X_train,y_train): 
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num = num + (X_train[i] - X_train.mean()) + (y_train[i] - y_train.mean())
            den = den + (X_train[i] - X_train.mean()*X_train[i] - X_train.mean())
            self.coef_ = num/den 
            self.intercept_ = y_train.mean() - (self.coef_*X_train.mean())
            
    def predict(self,X_test):
        return self.coef_ * X_test + self.intercept_ 
    
    
# Multiple Linear Regression for ND Data
class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        
        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_=betas[1:]
        
    def predict(self,X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred