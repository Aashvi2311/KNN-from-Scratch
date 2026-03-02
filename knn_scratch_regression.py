import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('regression_data.csv')
X = df[['X']].values
y = df['y'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def euclidean_distance(p,q):
        return np.sqrt((np.sum((p-q)**2)))

class KNN:

    def __init__(self,k):
        self.k = k
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        predictions = []

        for test_point in X_test:
            distances = []

            for i,train_point in enumerate(self.X_train):
                distance = euclidean_distance(train_point,test_point)
                distances.append([distance,self.y_train[i]])

            #sort distances
            distances.sort(key=lambda x: x[0])

            #find k nearest 
            k_nearest = distances[:self.k]
            values = [value for _,value in k_nearest]

            #find avg of k nearest
            avg_value = np.mean(values)
            predictions.append(avg_value)
        
        return np.array(predictions)
    

model_scratch = KNN(k=3)
model_scratch.fit(X_train,y_train)
y_pred_scratch = model_scratch.predict(X_test)


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test,y_test)
plt.scatter(X_test,y_pred_scratch)
plt.scatter(X_test,y_pred)
plt.show()
