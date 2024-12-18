import numpy as np 


# RidgeRegression - L2
class RideRegression:
    def __init__(self,alpha=0.1):
        self.coef_=None
        self.interncept_=None
        self.alpha=alpha
            
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        I = np.identity(X_train.shape[1])
        I[0][0] = 0
        
        result = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha * I).dot(X_train.T).dot(y_train)
        self.interncept_ = result[0]
        self.coef_ = result[1:]
        
    def predict(self,X_test):
        return np.dot(self.coef_,X_test) + self.interncept_