import numpy as np

def step(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class perceptron:
    def __init__(self, epochs=50, learning_rate=0.1):
        self.coef_=None
        self.intercept_=None
        self.epochs=epochs
        self.lr = learning_rate
        
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        weights = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            j = np.random.randint(X_train.shape[0])
            y_hat = step(np.dot(X_train[j],weights))
            weights = weights + self.lr *(y_train[j]-y_hat) * X_train[j]
            
        self.intercept_=weights[0]
        self.coef_=weights[1:]
        
    def predict(self,X_test):
        X_test = np.insert(X_test,0,1,axis=1)
        return step(np.dot(X_test, np.append(self.coef_,self.intercept_)))
    
class Logistic_Regression:
    def __init__(self,lr=0.001,epcohs=100):
        self.coef_=None
        self.intercept_=None
        self.weights = None
        self.lr=lr
        self.epochs=epcohs
        
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        self.weights = np.zeros(X_train.shape[1])
        
        for i in range(self.epochs):
            y_hat = sigmoid(np.dot(X_train,self.weights))
            self.weights=self.weights + self.lr*np.dot((y_train-y_hat),X_train)/X_train.shape[0]
            
        self.coef_=self.weights[1:]
        self.intercept_=self.weights[0]
        
    def predict(self,X_test):
        linear_pred = np.dot(X_test,self.coef_)+self.intercept_
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred    

    
