import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression




def showDataOnHistogram(data):
    plt.hist(data,bins=50)
    plt.show()

def showDataOnBarDiagram(x, y):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x, y)
    plt.show()

def transformToNormalDistribution(data):
    return data.transform([np.sqrt, np.exp, np.log, np.reciprocal])

# get training set
trainingData = pd.read_excel(r'WB.xls')

# get header labels
labels = trainingData.head()

# get features values
# 'Population', 'PopDensity', 'AgingRatio', 'ServicesHi', 'HealthServ', 'Landuse', 'Commercial', 'RoadDensit', 'GreenAreas', 'Open_spave'
features = (trainingData.iloc[:,range(5,15)])
population, populationDensity, agingRatio, servicesHierarchy, healthServices, landUse, commercial, roadDensity, greenAreas, openSpace = [trainingData.iloc[:,i] for i in range(5,15)]

selectedFeatures = ["HealthServ","ServicesHi","PopDensity"]

# get count corona cases
coronaCases = (trainingData.loc[:,"CORONA__Ca"])


# get regions names
regions = trainingData.loc[:,"NAMEEN"]




# print(MinMaxScaler().fit_transform((trainingData.loc[:, selectedFeatures]))[index])
# coronaCasesScaled = list(map(lambda x: x/100,coronaCases))
# populationScaled = list(map(lambda x: x/10000,population))
#
# print((trainingData.iloc[:, range(5,15)]).iloc[0])

# Visualize data
X = np.array([1, -4, 5, 6, -8, 5]) # here should be your X in np.array format

# print(MinMaxScaler().fit_transform((np.array(coronaCases))[:, np.newaxis]))

# print(MinMaxScaler(feature_range=(0, 1)).fit_transform([(trainingData.loc[:,"CORONA__Ca"])]))
# transformedCoronaCases = transformToNormalDistribution(coronaCases)
# print(transformedCoronaCases)

