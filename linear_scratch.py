import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('linear_data.csv')

#Implementing the model from scratch
def loss_function(a,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].X
        y = points.iloc[i].y
        total_error += (y - (a*x + b)) ** 2
    return total_error/float(len(points))

def gradient_descent(a_new,b_new,points,L):
    a_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].X
        y = points.iloc[i].y
        a_gradient += - (2/n) * x * (y - (a_new * x + b_new))
        b_gradient += - (2/n) * (y - (a_new * x + b_new))

    a = a_new - a_gradient*L
    b = b_new - b_gradient*L
    return a,b

a = 0
b = 0
L = 0.001
epochs = 1000

for i in range(epochs):
    a,b = gradient_descent(a,b,data,L)

#print(a,b)

#plt.scatter(data.X, data.y,color = 'black')
#plt.plot(data.X, a*data.X+b,color = 'red')
#plt.show()

X = data[['X']]
y = data[['y']]

#Split into training and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
y_pred_1 = a*X_test + b

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred_2 = model.predict(X_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred_1))
print(mean_squared_error(y_test,y_pred_2))

plt.scatter(X_test,y_test,color='black')
plt.plot(X_test,y_pred_1,color='red')
plt.plot(X_test,y_pred_2,color='green',linestyle='--')
plt.show()