import pandas as pd
from main import features, showCorrelation, coronaCases, trainingData, healthServices, servicesHierarchy, population, landUse, commercial, populationDensity, featuresSeries, boxPlot, getOutliers

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

# Show outliers using box plot for each feature
for feature in featuresSeries:
    boxPlot(feature, feature.name, "value")

# get tuples indices that have outliers on features
outliersIndicesOnFeatures = {}
for feature in featuresSeries:
    outliersIndicesOnFeatures[feature.name] = getOutliers(feature)
print(outliersIndicesOnFeatures)

# get tuple indices that have more than one feature has outlier
tuplesIndicesHaveOutlier = {}
for key, indices in outliersIndicesOnFeatures.items():
    for index in indices:
        if index in tuplesIndicesHaveOutlier:
            tuplesIndicesHaveOutlier[index] += 1
        else:
            tuplesIndicesHaveOutlier[index] = 1

