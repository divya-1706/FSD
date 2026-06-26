import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
x=iris.data
y=iris.target
target_names=iris.target_names
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
k_values=list(range(1,21))
accuracies=[]
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    accuracies.append(acc)
    print(f"k={k},accuracy={acc:2f}")
plt.figure(figsize=(8,6))
plt.plot(k_values,accuracies,marker='o',linestyle='-',color='b')
plt.title("k-NN accuracy on iris datasets")
plt.xlabel("number of neighbors(K)")
plt.ylabel("accuracy")
plt.grid(True)
plt.show()
