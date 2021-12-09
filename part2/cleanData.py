from main import trainingData, labels,\
    getFeaturesIndeicesHaveOutliers, getTuplesIndicesHaveOutliers

# Count missing values
print("Number of Missing values", trainingData.isna().sum().sum())

# Drop redundant tuples and get the number of duplicated records
numberOfOldRecords = len(trainingData)
trainingData.drop_duplicates(subset=list(labels)[4:15], keep=False, inplace=True)
print("Number of duplicated records", numberOfOldRecords - len(trainingData))
indicesTuplesToRemove = set()

# Check redundant data in each feature
for feature in [(trainingData.duplicated(keep=False, subset=[i, "Population"])) for i in ['NAMEAR', 'NAMEEN']]:
    for index, record in zip(range(len(feature)), feature):
        if record:
            print(index, feature.name, "it's duplicated error")
            indicesTuplesToRemove.add(index)

# Check if these features have value less or equal 0
countWrongRecords = 0
for featureAboveZero, featureAboveMinusOne in zip([trainingData.loc[:, i] for i in ["Population", "PopDensity", "ServicesHi", "Landuse", "Open_spave"]],\
                   [trainingData.loc[:, i] for i in ["AgingRatio", "HealthServ", "Commercial", "RoadDensit", "GreenAreas"]]):
    for index, record1, record2 in zip(range(len(featureAboveZero)), featureAboveZero, featureAboveMinusOne):
        if record1 <= 0:
            print(index, featureAboveZero.name, "Has value less than or equal zero")
            indicesTuplesToRemove.add(index)
            countWrongRecords += 1
        if record2 < 0:
            print(index, featureAboveZero.name, "Has value less than zero")
            indicesTuplesToRemove.add(index)
            countWrongRecords += 1
print("Number of records have zeros", countWrongRecords)

# remove tuples have wrong and duplicated data and export it
temporalData = trainingData.drop(indicesTuplesToRemove)
temporalData.to_excel("../WB-Cleaned.xls", index=False)
# Handle Noisy data by using smoothing by median
temporalTrainingData = temporalData.copy()
for label in list(labels)[4:15]:
    if label in ["Open_spave", "Landuse", "PopDensity", "ServicesHi"]:
        continue
    temporalTrainingData = temporalTrainingData.sort_values(by=[label])
    feature = list(temporalTrainingData.loc[:, label])
    temporal = feature.copy()
    for i in range(0, 175, 5):
        feature[i: i + 5] = [temporal[(i + 2) // 2] for j in range(5)]
    feature[180:182] = [temporal[(180 + 2) // 2] for j in range(2)]
    temporalTrainingData[label] = feature
temporalTrainingData.to_excel("CleanedTrainingDataAfterSmoothing.xls", index=False)

# get number of tuples have outliers
print("Number of tuples have outliers after smoothing", len(getTuplesIndicesHaveOutliers(getFeaturesIndeicesHaveOutliers(temporalTrainingData))))





