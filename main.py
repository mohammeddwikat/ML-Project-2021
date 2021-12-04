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

def boxPlot(data, xLabel="", yLabel=""):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.boxplot(data)
    plt.show()

def getOutliers(data):
    Q1 = np.percentile(data, 25, interpolation='midpoint')
    Q3 = np.percentile(data, 75, interpolation='midpoint')
    IQR = Q3 - Q1
    lowLimit = Q1 - 1.5 * IQR
    upLimit = Q3 + 1.5 * IQR
    countOutlieres = 0
    indices = []
    for key, i in zip(range(0, len(data)), data):
        if i > upLimit or i < lowLimit:
            countOutlieres += 1
            indices.append(key)
    print("Number of outliers in", data.name, countOutlieres)
    return indices

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
featuresSeries = [population, populationDensity, agingRatio, servicesHierarchy, healthServices, landUse, commercial, roadDensity, greenAreas, openSpace]

# get count corona cases
coronaCases = (trainingData.loc[:,"CORONA__Ca"])


# Steps
# - read data
# - normalize data
# - find the distribution of target and transform it to the normal distribution as much as possible
# - find the relation between variables and target
# - find the correlation between all features
# - use PCA
# - split the data to test and train
# - train the data using multi linear regression
# - test the model
# - evaluate the model using cross validation
#


