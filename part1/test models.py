from joblib import load
from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from main import coronaCases, features, trainingData, labels

DT_model = load("../Models/DecisionTreeModel.joblib")
reg = load("../Models/regressionModel.joblib")
knn = load("../Models/K-NearestModel.joblib")

# get test and training data
_, X_test, __, y_test = train_test_split(features, coronaCases, test_size=0.25)

# Models score using cross validation
scores = cross_val_score(DT_model, features, coronaCases, cv=5)
print("Decision Tree Score using cross validation:", scores.mean())
scores = cross_val_score(reg, features, coronaCases, cv=5)
print("Regression model Score using cross validation:", scores.mean())
scores = cross_val_score(knn, features, coronaCases, cv=5)
print("KNN regressor model Score using cross validation:", scores.mean())

# Models score using R^2
print("Decision Tree Score using R^2:", DT_model.score(X_test, y_test))
print("Regression model score using R^2:", reg.score(X_test, y_test))
print("KNN regressor model score using R^2:", knn.score(X_test, y_test))

# index = 64
# print((trainingData.iloc[:, range(0,5)]).iloc[index])
# print(DT_model.predict([((trainingData.loc[:, list(labels)[5:15]]).iloc[index])]))
# print(reg.predict([((trainingData.loc[:, list(labels)[5:15]]).iloc[index])]))

# coefficients for linear regression
print(reg.coef_)
print(reg.intercept_)

# Visualize the decision tree
iris = load_iris()
X, y = iris.data, iris.target
tree.plot_tree(DT_model)
plt.show()
