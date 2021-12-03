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

def showCorrelation(x, y, xLabel="", yLabel=""):
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def transformToNormalDistribution(data):
    return data.transform([np.sqrt, np.exp, np.log, np.reciprocal])

# get training set
trainingData = pd.read_excel(r'../WB.xls')

# get header labels
labels = trainingData.head()

# get regions names
regions = trainingData.loc[:,'NAMEEN']

# get features values
# 'Population', 'PopDensity', 'AgingRatio', 'ServicesHi', 'HealthServ', 'Landuse', 'Commercial', 'RoadDensit', 'GreenAreas', 'Open_spave'
features = (trainingData.iloc[:,range(5,15)])
population, populationDensity, agingRatio, servicesHierarchy, healthServices, landUse, commercial, roadDensity, greenAreas, openSpace = [trainingData.iloc[:,i] for i in range(5,15)]

# get count corona cases
coronaCases = (trainingData.loc[:,"CORONA__Ca"])

# show corona cases distribution
showDataOnHistogram(coronaCases, 190, "Number Of Cases", "Frequency")

# calculate the correlation between features
pd.set_option('display.max_columns', None)
print(features.corr(method="pearson"))
pd.set_option('display.max_columns', 5)

# show relation between features that have high correlation
showCorrelation(healthServices, servicesHierarchy, "healthServices", "servicesHierarchy")
showCorrelation(population, healthServices, "population", "healthServices")
showCorrelation(population, servicesHierarchy, "population", "servicesHierarchy")

# show examples of features have no relation between them
showCorrelation(landUse, commercial, "landUse", "commercial")
showCorrelation(populationDensity, population, "populationDensity", "population")

# show relation and compute correlation between each feature and target.
for feature in [trainingData.iloc[:,i] for i in range(5,15)]:
    print("correlation {feature} with corona cases: {correlation}".format(feature=feature.name, correlation=feature.corr(coronaCases)))
    showCorrelation(feature, coronaCases, feature.name, "coronaCases")



