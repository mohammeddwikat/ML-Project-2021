import pandas as pd
from main import features, labels, showCorrelation, coronaCases, trainingData, healthServices,\
    servicesHierarchy, population, landUse, commercial, populationDensity, boxPlot, getOutliers, euclideanDistance,\
    getFeaturesIndeicesHaveOutliers, getTuplesIndicesHaveOutliers, showDataOnHistogram
import xlsxwriter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




# show corona cases distribution
showDataOnHistogram(coronaCases, 190, "Number Of Cases", "Frequency")

# show features distribution
for i in [trainingData.iloc[:,j] for j in range(5,15)]:
    showDataOnHistogram(i, 191, i.name, "Frequency")

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

# heat map describe correlation between variables
ax = sns.heatmap((trainingData.iloc[:, range(4, 15)]).corr(), annot=True, vmin=0, vmax=1)

# Show outliers using box plot for each feature
for feature in [trainingData.iloc[:,i] for i in range(4,15)]:
    boxPlot(feature, feature.name, "value")

# get indices data that have outliers on features
outliersIndicesOnFeatures = getFeaturesIndeicesHaveOutliers(trainingData)
print(outliersIndicesOnFeatures)

# get tuples indices that have outliers
print("Number of Tuples have outliers", len(getTuplesIndicesHaveOutliers(outliersIndicesOnFeatures)))

# Principle component Analysis
pca = PCA()
components = pca.fit_transform(trainingData.iloc[:,range(5, 15)])
eigenValues = pca.explained_variance_
print("The percentage of coverage using Y1 and Y2",sum(eigenValues[0:2]) / sum(eigenValues))

components_df = pd.DataFrame(data = components, columns = ["y"+str(i) for i in range(1, len(eigenValues)+1)])
sns.scatterplot(components_df['y1'], components_df['y2'], c=coronaCases, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
plt.show()

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

