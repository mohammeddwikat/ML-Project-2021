from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from main import showDataOnHistogram, trainingDataCleaned, transformToNormalDistribution
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = trainingDataCleaned.iloc[:, [5, 8, 9]]
coronaCases = trainingDataCleaned.loc[:, ["CORONA__Ca"]]
features_scaled = scaler.fit_transform(features)
coronaCasesTransformedToNormal = transformToNormalDistribution(coronaCases)

# Discretization  Transforms
kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
showDataOnHistogram(coronaCasesTransformedToNormal, len(coronaCases))

coronaCases_transformed = kbins.fit_transform(transformToNormalDistribution(coronaCases))
showDataOnHistogram(coronaCases_transformed, 5, "Corona cases categories")

pca = PCA()
components = pca.fit_transform(features_scaled)
eigenValues = pca.explained_variance_
print("The percentage of coverage using Y1 and Y2", sum(eigenValues[0:2]) / sum(eigenValues))

components_df = pd.DataFrame(data=components, columns=["y"+str(i) for i in range(1, len(eigenValues)+1)])
sns.scatterplot(components_df['y1'], components_df['y2'], c=coronaCases_transformed, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 5))
plt.show()

labels = ["Safe", "Low", "Middle", "High", "Dangerous"]
coronaCases_transformed_categories = pd.Series(list(map(lambda index: labels[int(index)], coronaCases_transformed)))
neigh = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, coronaCases_transformed_categories, test_size=0.3, random_state=0)
neigh.fit(X_train, y_train)
print("The accuracy for KNN calssifier using cross validation", cross_val_score(neigh, features_scaled, coronaCases_transformed_categories, cv=5).mean())
cm = confusion_matrix(neigh.predict(features_scaled), coronaCases_transformed_categories)
print("The accuracy for KNN classifier using confusion matrix", neigh.score(X_test, y_test))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

