from joblib import load
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import part1.analyseData as analyseData
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

DT_model = load("../Models/DecisionTreeModel.joblib")
reg = load("../Models/regressionModel.joblib")

# get test and training data
_, X_test, _, y_test = train_test_split(analyseData.features, analyseData.coronaCases, test_size=0.25, random_state=0)

# Models score using cross validation
scores = cross_val_score(DT_model, analyseData.features, analyseData.coronaCases, cv=5)
print("Decision Tree Score using cross validation:", sum(scores)/len(scores))
scoresReg = cross_val_score(reg, analyseData.features, analyseData.coronaCases, cv=5)
print("Regression model Score using cross validation:", sum(scoresReg)/len(scoresReg))

# Models score using R^2
print("Decision Tree Score using R^2:", DT_model.score(X_test, y_test))
print("Regression model score using R^2:", reg.score(X_test, y_test))


iris = load_iris()
X, y = iris.data, iris.target
tree.plot_tree(DT_model)
plt.show()
