from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import joblib
from main import features, coronaCases
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


# split data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, coronaCases, test_size=0.25)

# Decision Tree Predictor Model
DT_model = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
DT_predict = DT_model.predict(X_test)
print("Decision Tree Score using R^2:", DT_model.score(X_test, y_test))
print("Decision Tree Score using cross validation", cross_val_score(DT_model, features, coronaCases, cv=5).mean())

# Linear Regression
reg = LinearRegression().fit(X_train, y_train)
y_predict = reg.predict(X_test)
print("Regression model score using R^2:", reg.score(X_test, y_test))
print("Regression model Score using cross validation", cross_val_score(reg, features, coronaCases, cv=5).mean())

# K-Nearest Regressor
neigh = KNeighborsRegressor(n_neighbors=11)
neigh.fit(X_train, y_train)
print("K-Nearest Regressor model score using R^2:", cross_val_score(neigh, features, coronaCases, cv=5).mean())
print("K-Nearest Regressor model score using cross validation:", neigh.score(X_test, y_test))

# Save models
filename = "regressionModel.joblib"
joblib.dump(reg, filename)
filename = "DecisionTreeModel.joblib"
joblib.dump(DT_model, filename)
filename = "K-NearestModel.joblib"
joblib.dump(neigh, filename)