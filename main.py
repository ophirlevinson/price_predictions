import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt



class Data():
    def __init__(self,fname='AAPL.csv',batch_size=10):
        self.price = np.array(pd.read_csv(fname).loc[:,'Close'])
        self.date = np.array([x for x in range(0,len(self.price))])#np.array(pd.read_csv(fname).loc[:,'Date'])
        self.batch_size = batch_size
        self.linear_regresion_pred = []
        self.linear_ridge_pred = []
        self.linear_lasso_pred = []
    
    def get(self,indices): # returns dates and prices of particular indices
        return np.array(self.date[indices]), np.array(self.price[indices])
    
    def generate(self): #generate a date and price iterator of batch_size amounts
        for i in range(0,len(self.date)-self.batch_size):
            yield(self.get([index for index in range(i,i+self.batch_size)]))
    
    def predict(self,date,price,type): #performs linear regression prediction for the next price
        if type == 'linear':
            model = linear_model.LinearRegression()
        if type == 'ridge':
            model = linear_model.Ridge(alpha=0.3)
        if type == 'lasso':
            model = linear_model.Lasso(alpha=0.1)
        reg = model.fit(date.reshape(-1, 1),price)
        pred = reg.predict(np.array(date[len(date)-1]+1).reshape(-1,1))
        return pred[0]
    def plot(self):
        plt.figure()
        # Linear Regression
        ax = plt.subplot(4,1,1)
        
        plt.ylabel('Price')
        plt.title('LinearRegression (batch='+str(self.batch_size)+')')
        plt.plot(self.price[data.batch_size:],'b')
        plt.plot(np.array(self.linear_regresion_pred),'g')
        ax.set_xticklabels([])
        
        #Linear Rigde
        ax = plt.subplot(4,1,2)
        plt.ylabel('Price')
        plt.title('Ridge (batch='+str(self.batch_size)+')')
        plt.plot(self.price[data.batch_size:],'b')
        plt.plot(np.array(self.linear_ridge_pred),'g')
        ax.set_xticklabels([])
        
        #Lasso Rigde
        ax = plt.subplot(4,1,3)
        plt.ylabel('Price')
        plt.title('Lasso (batch='+str(self.batch_size)+')')
        plt.plot(self.price[data.batch_size:],'b')
        plt.plot(np.array(self.linear_lasso_pred),'g')
        ax.set_xticklabels([])
        
        #Difference between regressions
        ax = plt.subplot(4,1,4)
        plt.title('Differece')
        plt.ylabel('Difference')
        plt.plot(np.array(self.linear_lasso_pred) - np.array(self.linear_regresion_pred),'g')
        plt.plot(np.array(self.linear_ridge_pred) - np.array(self.linear_regresion_pred),'r')
        ax.set_xticklabels([])

data = Data(batch_size=60)

for batch_data in Data().generate():
    date, price = batch_data
    data.linear_regresion_pred.append(data.predict(date,price,'linear'))
    data.linear_ridge_pred.append(data.predict(date,price,'ridge'))
    data.linear_lasso_pred.append(data.predict(date,price,'lasso'))
    
data.plot()


