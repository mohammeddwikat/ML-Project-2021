import pandas as pd
from main import features, showCorrelation, coronaCases, trainingData, healthServices,\
    servicesHierarchy, population, landUse, commercial, populationDensity, boxPlot, getOutliers, euclideanDistance,\
    getFeaturesIndeicesHaveOutliers, getTuplesIndicesHaveOutliers
import xlsxwriter

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
for feature in [trainingData.iloc[:,i] for i in range(4,15)]:
    boxPlot(feature, feature.name, "value")

# get indices data that have outliers on features
outliersIndicesOnFeatures = getFeaturesIndeicesHaveOutliers(trainingData)
print(outliersIndicesOnFeatures)

# get tuples indices that have outliers
print("Number of Tuples have outliers", len(getTuplesIndicesHaveOutliers(outliersIndicesOnFeatures)))

# find the dissimilarity between tuples
table = []
for index, tuple1 in features.iterrows():
    row = [index]
    for _, tuple2 in features.iterrows():
        row.append(euclideanDistance(tuple1, tuple2))
    table.append(row)

# export the table on excel sheet
workbook = xlsxwriter.Workbook('dissimilarityMatrix.xlsx')
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(table):
    worksheet.write_column(row, col, data)
workbook.close()