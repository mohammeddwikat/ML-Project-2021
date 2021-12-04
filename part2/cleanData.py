from main import trainingData, featuresSeries, coronaCases, labels

# Count missing values
countMissingValues = 0
for feature in featuresSeries:
    for value in feature:
        countMissingValues += 1 if value is None else 0
for case in coronaCases:
    countMissingValues += 1 if case is None else 0
    
print("Number of Missing values", countMissingValues)

# Drop redundant tuples and get the number of duplicated records
numberOfOldRecords = len(trainingData)
trainingData.drop_duplicates(subset=list(labels)[4:15], keep=False, inplace=True)
print("Number of duplicated records", numberOfOldRecords - len(trainingData))


# some tuples have incorrect population
# remove tuples have corona cases higher than population
# according to WHO and correlation between aging and corona cases remove aging feature
# compare between green and open areas
# remove data that have service hierarchy 0 or change it to 1
# eliminate Commercial services because the population is actor in computing it
#
