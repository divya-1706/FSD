from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

boston=fetch_california_housing()
x=boston.data
y=boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
lr=LinearRegression()
lr.fit(x_train_scaled, y_train)
y_pred=lr.predict(x_test_scaled)
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train_scaled)
x_test_poly=poly.transform(x_test_scaled)
pr=LinearRegression()
pr.fit(x_train_poly, y_train)
y_pred_pr=pr.predict(x_test_poly)
ridge=Ridge(alpha=1.0)
ridge.fit(x_train_scaled, y_train)
y_pred_ridge=ridge.predict(x_test_scaled)
lasso=Lasso(alpha=0.1)
lasso.fit(x_train_scaled, y_train)
y_pred_lasso=lasso.predict(x_test_scaled)
def evaluate(name,y_tree,y_pred):
    print(f"{name}:")
    print(f"MSE:{mean_squared_error(y_tree,y_pred):.2f}")
    print(f"R2:{r2_score(y_tree,y_pred):.2f}")
evaluate("Linear Regression",y_test,y_pred)
evaluate("Polynomial Regression",y_test,y_pred_pr)
evaluate("Ridge Regression",y_test,y_pred_ridge)
evaluate("Lasso Regression",y_test,y_pred_lasso)

