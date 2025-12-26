import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

data = pd.read_csv('linear_data.csv')

#Train test split data before scaling
X = data[['X']]
y = data['y']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#KNN requires feature scaling because it is distance based
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Xtrain,Xtest,ytrain are Pandas dataframe. Need to convert to array to iterate
X_train_arr = scaler.fit_transform(X_train.values)
X_test_arr = scaler.transform(X_test.values) #Scaling only on training data
y_train_arr = y_train.values

k = 3
neighbors_all = []

for x_test in X_test_arr:
    distances = []
    for i in range(len(X_train_arr)):
        x_train = X_train_arr[i]
        y_value = y_train_arr[i]

        #Compute distance between one test point and all train points
        distance = euclidean_distance(x_test,x_train)
        #Store distance and y_train value for each point
        distances.append((distance,y_value)) #(distance from test point, y value for train point) 

    #Sort by distance based on X[0] distance part
    distances.sort(key=lambda x:x[0]) 

    #Take first K neighbors distance -- k smallest distance
    k_neighbors = distances[:k]

    #Majority vote in Classification - Count how many belong to each class and assign with highest majority
    #class_count = {}
    #for _,label in k_neighbors:
    #    class_count[label] = class_count.get(label,0) + 1

    #Ignores distance value, finds y target values, finds its sum and divides by number of neighbors
    prediction = sum(y for _, y in k_neighbors)/k
    #prediction_classification = max(class_count,key=class_count.get)

    #Store predictions
    neighbors_all.append(prediction)

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train_arr,y_train_arr)
y_pred = model.predict(X_test_arr)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,neighbors_all))
print(mean_squared_error(y_test,y_pred))

# Flatten everything - To convert all data in 1D to plot in Matplotlib
X_test_flat = X_test_arr.ravel()
neighbors_all_arr = np.array(neighbors_all).ravel()
y_pred_arr = y_pred.ravel()
y_test_arr = y_test.values.ravel()

# Sort by X for smooth line
sorted_idx = np.argsort(X_test_flat)
X_sorted = X_test_flat[sorted_idx]
y_test_sorted = y_test_arr[sorted_idx]
manual_knn_sorted = neighbors_all_arr[sorted_idx]
sklearn_knn_sorted = y_pred_arr[sorted_idx]

plt.scatter(X_sorted, y_test_sorted, color='black',alpha=0.6)
plt.plot(X_sorted, manual_knn_sorted, color='red',linewidth=3)
plt.plot(X_sorted, sklearn_knn_sorted, color='green',linewidth=2)
plt.show()




