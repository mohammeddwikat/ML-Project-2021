from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import joblib
from main import features, coronaCases

# split data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, coronaCases, test_size=0.25, random_state=0)

# Decision Tree Predictor Model
DT_model = DecisionTreeRegressor().fit(X_train, y_train)
DT_predict = DT_model.predict(X_test)
print("Decision Tree Score using R^2:", DT_model.score(X_test, y_test))

# Linear Regression
reg = LinearRegression().fit(X_train, y_train)
y_predict = reg.predict(X_test)
print("Regression model score using R^2:", reg.score(X_test, y_test))


filename = "regressionModel.joblib"
joblib.dump(reg, filename)
filename = "DecisionTreeModel.joblib"
joblib.dump(DT_model, filename)