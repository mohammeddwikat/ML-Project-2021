from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

trainingData = pd.read_excel(r'trainingDataAfterHandling.xls')
features = trainingData.iloc[:, range(5, 15)]
coronaCases = trainingData.loc[:, "CORONA__Ca"]

# split data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, coronaCases, test_size=0.25)

# K-Nearest Regressor
neigh = KNeighborsRegressor(n_neighbors=11)
neigh.fit(X_train, y_train)
print("K-Nearest Regressor model score using cross validation:", cross_val_score(neigh, features, coronaCases, cv=5).mean())
print("K-Nearest Regressor model score using using R^2:", neigh.score(X_test, y_test))







