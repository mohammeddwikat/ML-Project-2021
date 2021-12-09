from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from main import labels, transformToNormalDistribution
from itertools import combinations
import pandas as pd
from sklearn.preprocessing import StandardScaler

# trainingData, features, coronaCases = pd
trainingData = pd.read_excel(r'../WB-Cleaned.xls')
coronaCases = transformToNormalDistribution(trainingData.loc[:,list(labels)[4]])
features = trainingData.loc[:,list(labels)[5:15]]
featuresNames = list(labels)[5:15]

combinationsFeatures = []
for r in range(1, 11):
    combinationsFeatures.append(list(combinations(featuresNames, r)))

scores = []
for i in combinationsFeatures:
    for selectedFeatures in i:
        x_train, _, y_train, __ = train_test_split(features.loc[:, selectedFeatures], coronaCases, test_size=0.25)
        x_train_scaled = StandardScaler().fit_transform(x_train)
        neigh = KNeighborsRegressor(n_neighbors=7).fit(x_train, y_train)
        scores.append(cross_val_score(neigh, StandardScaler().fit_transform(features.loc[:, selectedFeatures]), coronaCases, cv=5).mean())
        print("K-Nearest Regressor model score using cross validation:", scores[-1])
        print(selectedFeatures)
        print("-" * 50)

print(max(scores))

