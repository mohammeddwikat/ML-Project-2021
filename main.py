import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer




def showDataOnHistogram(data):
    plt.hist(data,bins=50)
    plt.show()

def transformToNormalDistribution(data):
    return data.transform([np.sqrt, np.exp, np.log, np.reciprocal])

# get training set
trainingData = pd.read_excel(r'WB.xls')

# get header labels
labels = trainingData.head()
# print(list(labels))

# get features values
# 'Population', 'PopDensity', 'AgingRatio', 'ServicesHi', 'HealthServ', 'Landuse', 'Commercial', 'RoadDensit', 'GreenAreas', 'Open_spave'
features = (trainingData.iloc[:,range(5,15)])
population, populationDensity, agingRatio, servicesHierarchy, healthServices, landUse, commercial, roadDensity, greenAreas, openSpace = [trainingData.iloc[:,i] for i in range(5,15)]
#print(features)
selectedFeatures = ["HealthServ","ServicesHi","PopDensity"]

# get count corona cases
coronaCases = (trainingData.loc[:,"CORONA__Ca"])
# print(coronaCases)


# get regions names
regions = trainingData.loc[:,"NAMEEN"]

# Scale date [0 - 1] and split it to train and test sets
X_train, X_test, y_train, y_test = train_test_split(features[selectedFeatures], coronaCases, test_size=0.25, random_state=0)
X_train_scaled, X_test_scaled = [MinMaxScaler().fit_transform(i) for i in [X_train, X_test]]


# Decision Tree Predictor Model
DT_model = DecisionTreeRegressor().fit(X_train_scaled, y_train)
DT_predict = DT_model.predict(X_test_scaled)
# print(*DT_predict)
print(DT_model.score(X_test_scaled, y_test))

# print(DT_model.predict(scaler.transform(pd.DataFrame({"Population":[905], "PopDensity":[6.08], "AgingRatio":[0.02], "ServicesHi":[1], "HealthServ":[0], "Landuse":[2.555], "Commercial":[0], "RoadDensit":[0.193], "GreenAreas":[0.011],"Open_spave": [0.831]}))))
# print(scaler.transform([905, 6.08, 0.02, 1, 0, 2.555, 0, 0.193, 0.011, 0.831]))
# print(pd.DataFrame([905, 6.08, 0.02, 1, 0, 2.555, 0, 0.193, 0.011, 0.831]))
#print(scaler.transform(pd.DataFrame({"Population":[905], "PopDensity":[6.08], "AgingRatio":[0.02], "ServicesHi":[1], "HealthServ":[0], "Landuse":[2.555], "Commercial":[0], "RoadDensit":[0.193], "GreenAreas":[0.011],"Open_spave": [0.831]})))

index = 138
print((trainingData.iloc[:, range(0,5)]).iloc[index])
print(DT_model.predict([MinMaxScaler().fit_transform((trainingData.loc[:, selectedFeatures]))[index]]))


# Linear Regression
reg = LinearRegression().fit(X_train_scaled, y_train)


y_predict = reg.predict(X_test_scaled)
# = reg.predict([min_max_scaler.fit_transform((trainingData.loc[:, selectedFeatures]))[index]])



print(reg.score(X_test_scaled, y_test))
print(reg.predict([MinMaxScaler().fit_transform((trainingData.loc[:, selectedFeatures]))[index]]))

# print(MinMaxScaler().fit_transform((trainingData.loc[:, selectedFeatures]))[index])
coronaCasesScaled = list(map(lambda x: x/100,coronaCases))
# populationScaled = list(map(lambda x: x/10000,population))
#
# print((trainingData.iloc[:, range(5,15)]).iloc[0])

# Visualize data
X = np.array([1, -4, 5, 6, -8, 5]) # here should be your X in np.array format

# print(MinMaxScaler().fit_transform((np.array(coronaCases))[:, np.newaxis]))

# print(MinMaxScaler(feature_range=(0, 1)).fit_transform([(trainingData.loc[:,"CORONA__Ca"])]))
transformedCoronaCases = transformToNormalDistribution(coronaCases)
print(transformedCoronaCases)
showDataOnHistogram(transformedCoronaCases['sqrt'])
# plt.plot(MinMaxScaler().fit_transform((np.array(population))[:, np.newaxis]), MinMaxScaler().fit_transform((np.array(coronaCases))[:, np.newaxis]), 'o')
# plt.show()


#pd.DataFrame({"Population":[905], "PopDensity":[6.08], "AgingRatio":[0.02], "ServicesHi":[1], "HealthServ":[0], "Landuse":[2.555], "Commercial":[0], "RoadDensit":[0.193], "GreenAreas":[0.011],"Open_spave": [0.831]})



# Steps
# - read data
# - find the distribution of target and transform it to the normal distribution as much as possible
# - find the relation between variables and target
# - find the correlation between all features
# - split the data to test and train
# - train the data using multi linear regression
# - test the model
# - evaluate the model using cross validation
#
