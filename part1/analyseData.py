import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def showDataOnHistogram(data, bins, xTitle="", yTitle=""):
    plt.hist(data, bins=bins)
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.show()

def showDataOnBarDiagram(xData, yData, xTitle, yTitle, ):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    ax.bar(xData, yData)
    plt.show()

def transformToNormalDistribution(data):
    return data.transform([np.sqrt, np.exp, np.log, np.reciprocal])

def showCorrelation(x, y, xLabel="", yLabel=""):
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

# get training set
trainingData = pd.read_excel(r'../WB.xls')
trainingData = trainingData.sort_values(by=['CORONA__Ca'])

# get header labels
labels = trainingData.head()

# get regions names
regions = trainingData.loc[:,'NAMEEN']

# get features values
# 'Population', 'PopDensity', 'AgingRatio', 'ServicesHi', 'HealthServ', 'Landuse', 'Commercial', 'RoadDensit', 'GreenAreas', 'Open_spave'
features = (trainingData.iloc[:,range(5,15)])
population, populationDensity, agingRatio, servicesHierarchy, healthServices, landUse, commercial, roadDensity, greenAreas, openSpace = [trainingData.iloc[:,i] for i in range(5,15)]
print(features)

# get count corona cases
coronaCases = (trainingData.loc[:,"CORONA__Ca"])

# show corona cases distribution
#showDataOnHistogram(coronaCases, 190, "Number Of Cases", "Frequency")

# calculate the correlation between features
pd.set_option('display.max_columns', None)
print(features.corr(method="pearson"))
pd.set_option('display.max_columns', 5)

# showCorrelation(populationDensity, population, "populationDensity", "population")
# showCorrelation(healthServices, servicesHierarchy, "healthServices", "servicesHierarchy")
# showCorrelation(landUse, commercial, "landUse", "commercial")
# showCorrelation(greenAreas, populationDensity, "greenAreas", "populationDensity")
#
for feature in [trainingData.iloc[:,i] for i in range(5,15)]:
    print("correlation {feature} with corona cases: {correlation}".format(feature=feature.name, correlation=feature.corr(coronaCases)))
    showCorrelation(feature, coronaCases, feature.name, "coronaCases")