import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
iris=load_iris()
x=pd.DataFrame(iris.data,columns=iris.feature_names)
y=pd.Series(iris.target,name='target')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print("Random Forest Classifier (Default Parameters):")
print("Accuracy:", accuracy_score(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
xticklabels=iris.target_names, yticklabels=iris.target_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
param_grid = {
'n_estimators': [10, 50, 100],
'max_depth': [None, 3, 5, 10],
'min_samples_split': [2, 4],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(x_train, y_train)
print("Best Parameters:", grid.best_params_)
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(x_test)
print("\nRandom Forest Classifier (Tuned Parameters):")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
importances = pd.Series(best_rf.feature_importances_, index=x.columns)
importances.sort_values().plot(kind='barh', title='Feature Importances')
plt.xlabel("Importance Score")
plt.show()