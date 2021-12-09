from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from main import trainingDataCleaned, transformToNormalDistribution, trainingDataManual
from sklearn.preprocessing import StandardScaler

# Standard Scaler to transform data
scaler = StandardScaler()

print("Model using smoothed data")
trainingData = trainingDataManual
features = trainingData.iloc[:, [5,8,9]]
coronaCases = trainingData.loc[:, "CORONA__Ca"]


# split data to training and testing sets
features_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, coronaCases, test_size=0.25)


# K-Nearest Regressor
neigh = KNeighborsRegressor(n_neighbors=11)
neigh.fit(X_train, y_train)
print("K-Nearest Regressor model score using cross validation:", cross_val_score(neigh, features_scaled, coronaCases, cv=5).mean())
print("K-Nearest Regressor model score using using R^2:", neigh.score(X_test, y_test))



# K-Nearest Regressor using cleaned data without smoothing
print("Model without using smoothed data")

features = trainingDataCleaned.iloc[:, [5, 8, 9]]
coronaCases = trainingDataCleaned.loc[:, ["CORONA__Ca"]]
features_scaled = scaler.fit_transform(features)
coronaCasesTransformedToNormal = transformToNormalDistribution(coronaCases)

# split data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, coronaCasesTransformedToNormal, test_size=0.25)

# K-Nearest Regressor
neigh = KNeighborsRegressor(n_neighbors=7)
neigh.fit(X_train, y_train)
print("K-Nearest Regressor model score using cross validation:", cross_val_score(neigh, features_scaled, coronaCasesTransformedToNormal, cv=5).mean())
print("K-Nearest Regressor model score using using R^2:", neigh.score(X_test, y_test))



