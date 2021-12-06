from scipy.stats import ttest_ind
from main import trainingDataCleanedManually

coronaCases = trainingDataCleanedManually.iloc[:, [4]]
features = trainingDataCleanedManually.iloc[:, range(5, 15)]

for feature in [trainingDataCleanedManually.iloc[:, i] for i in range(5,15)]:
    res = ttest_ind(feature, coronaCases)
    print(feature.name, res)