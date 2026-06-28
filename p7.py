import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
np.random.seed(42)
n=1000
data=pd.DataFrame({
    'Tenure':np.random.randint(1,72,n),
    'MonthlyCharges':np.random.uniform(20,120,n),
    'TotalCharges':lambda df: df['Tenure'] * df['MonthlyCharges'],
    'SupportCalls':np.random.poisson(2,n),
    'StreamingService':np.random.choice([0,1],size=n),
    'PaperlessBilling':np.random.choice([0,1],size=n),
    'Contact':np.random.choice([0,1,2],size=n),
})
data['TotalCharges']=data['Tenure']*data['MonthlyCharges']
data['churn']=((data['Tenure']<12)&(data['SupportCalls']>3)).astype(int)
x=data.drop('churn',axis=1)
y=data['churn']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier(max_depth=5,random_state=42)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("accuracy score:",accuracy_score(y_test,y_pred))
print("classification report:",classification_report(y_test,y_pred))
plt.figure(figsize=(18,10))
plot_tree(clf,feature_names=x.columns,class_names=['no churn','churn'],filled=True)
plt.title("Decision Tree for Customer Churn Prediction")
plt.show()
importances=pd.Series(clf.feature_importances_,index=x.columns)
importances.sort_values(ascending=False).plot(kind='bar',title='Feature Importances',figsize=(8,4))
plt.ylabel('Importance Score')
plt.show()