import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
np.random.seed(42)
n_samples=1000
age=np.random.randint(18,70,n_samples)
income=np.random.randint(20000,150000,n_samples)
browsing_time=np.random.normal(5,2,n_samples)
clicked=(0.3*(age<30)+0.4*(income<50000)+0.5*(browsing_time>5)+np.random.normal(0,0.2,n_samples))>0.8
clicked=clicked.astype(int)
df=pd.DataFrame({
    'Age':age,
    'Annual_Income':income,
    'Browsing_Hours':browsing_time,
    'Clicked_on_Ad':clicked
})
x=df[['Age','Annual_Income','Browsing_Hours']]
y=df['Clicked_on_Ad']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("classification report:")
print(classification_report(y_test,y_pred))
print("accuracy score:",accuracy_score(y_test,y_pred))