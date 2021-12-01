import part1.analyseData as analyseData
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import joblib

# split it to train and test sets
X_train, X_test, y_train, y_test = train_test_split(analyseData.features, analyseData.coronaCases, test_size=0.25, random_state=0)

# Decision Tree Predictor Model
DT_model = DecisionTreeRegressor().fit(X_train, y_train)
DT_predict = DT_model.predict(X_test)
scores = cross_val_score(DT_model, analyseData.features, analyseData.coronaCases, cv=5)
print("Decision Tree Score:",sum(scores)/len(scores))

# # Linear Regression
# reg = LinearRegression().fit(X_train, y_train)
# y_predict = reg.predict(X_test)
# scores = cross_val_score(reg, analyseData.features, analyseData.coronaCases, cv=5)
# print("Regression model score:",sum(scores)/len(scores))
#
# filename = "regressionModel.joblib"
# joblib.dump(reg, filename)

filename = "DecisionTreeModel.joblib"
joblib.dump(DT_model, filename)