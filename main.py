import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression



# plt.plot(MinMaxScaler().fit_transform((np.array(population))[:, np.newaxis]), MinMaxScaler().fit_transform((np.array(coronaCases))[:, np.newaxis]), 'o')
# plt.show()


#pd.DataFrame({"Population":[905], "PopDensity":[6.08], "AgingRatio":[0.02], "ServicesHi":[1], "HealthServ":[0], "Landuse":[2.555], "Commercial":[0], "RoadDensit":[0.193], "GreenAreas":[0.011],"Open_spave": [0.831]})



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
