from joblib import load
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import part1.analyseData as analyseData
from sklearn.model_selection import train_test_split


DT_model = load("../Models/DecisionTreeModel.joblib")
reg = load("../Models/regressionModel.joblib")

# get test data
_, X_test, _, y_test = train_test_split(analyseData.features, analyseData.coronaCases, test_size=0.25, random_state=0)

# Decision Tree score
print("Decision Tree Score:", DT_model.score(X_test, y_test))
print("Regression model score:",reg.score(X_test, y_test))


iris = load_iris()
X, y = iris.data, iris.target
tree.plot_tree(DT_model)
plt.show()
