#Simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('C:/Users/Asus/Desktop/booji.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/4,random_state=20)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.plot('Salary vs experience (training_set')
plt.xlabel('interest rate')
plt.ylabel('home price ')
plt.show()

plt.scatter(X_train, y_test,color ='red' )
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs experincce(Test set)')
plt.xlabel('interest rate')
plt.ylabel('home price')
plt.show()




