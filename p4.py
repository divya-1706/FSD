import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
housing=fetch_california_housing()
x=pd.DataFrame(housing.data,columns=housing.feature_names)
y=pd.Series(housing.target,name='MedHouseVal')
x_selected=x[['MedInc','HouseAge','AveRooms']]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x_selected)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("mean_squared_error:",mean_squared_error(y_test,y_pred))
print("r2_score:",r2_score(y_test,y_pred))
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,edgecolor='blue',alpha=0.7)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],'r--')
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("california-housing-actual VS predicted price")
plt.grid(True)
plt.show()