from main import trainingData, coronaCases, labels,\
    getFeaturesIndeicesHaveOutliers, getTuplesIndicesHaveOutliers
import pandas as pd

# Count missing values
countMissingValues = 0
for feature in [trainingData.iloc[:, i] for i in range(4, 15)]:
    for value in feature:
        countMissingValues += 1 if value is None else 0
for case in coronaCases:
    countMissingValues += 1 if case is None else 0

print("Number of Missing values", countMissingValues)

# Drop redundant tuples and get the number of duplicated records
numberOfOldRecords = len(trainingData)
trainingData.drop_duplicates(subset=list(labels)[4:15], keep=False, inplace=True)
print("Number of duplicated records", numberOfOldRecords - len(trainingData))
temporalTrainingData = trainingData.copy()

# Handle Noisy data by using smoothing by median
for label in list(labels)[4:15]:
    if label in ["Open_spave", "Landuse", "ServicesHi", "PopDensity"]:
        continue
    temporalTrainingData = temporalTrainingData.sort_values(by=[label])
    feature = list(temporalTrainingData.loc[:, label])
    temporal = feature.copy()
    for i in range(0, 187, 5):
        feature[i: i + 5] = [temporal[(i + 2) // 2] for j in range(5)]
    feature[190] = feature[188]
    temporalTrainingData[label] = feature
temporalTrainingData.to_excel("trainingDataAfterHandling.xlsx")

# get number of tuples have outliers
print("Number of tuples have outliers after smoothing", len(getTuplesIndicesHaveOutliers(getFeaturesIndeicesHaveOutliers(temporalTrainingData))))

# some tuples have incorrect population
# remove tuples have corona cases higher than population
# according to WHO and correlation between aging and corona cases remove aging feature
# compare between green and open areas
# remove data that have service hierarchy 0 or change it to 1
# eliminate Commercial services because the population is actor in computing it
#


