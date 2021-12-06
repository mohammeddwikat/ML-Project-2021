from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, chi2
from main import labels
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# trainingData, features, coronaCases = pd
trainingData = pd.read_excel(r'../WB-handledManually.xls')
coronaCases = trainingData.loc[:,list(labels)[4]]
features = trainingData.loc[:,list(labels)[5:15]]
featuresNames = list(labels)[5:15]
#
# combinationsFeatures = []
# for r in range(1, 11):
#     combinationsFeatures.append(list(combinations(featuresNames, r)))
#
# scores = []
# for i in combinationsFeatures:
#     for selectedFeatures in i:
#         X_train, X_test, y_train, y_test = train_test_split(features.loc[:, selectedFeatures], coronaCases, test_size=0.25)
#         neigh = KNeighborsRegressor(n_neighbors=11)
#         neigh.fit(X_train, y_train)
#         scores.append(cross_val_score(neigh, features.loc[:, selectedFeatures], coronaCases, cv=5).mean())
#         print("K-Nearest Regressor model score using cross validation:", scores[-1])
#         print(selectedFeatures)
#         print("-" * 50)
#
# print(max(scores))

