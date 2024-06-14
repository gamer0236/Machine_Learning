import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler 
from sklearn.datasets import load_iris

dataset = load_iris()
print(dir(dataset))
dataframe = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(dataframe)
dataframe.drop(["sepal length (cm)","sepal width (cm)"],axis="columns",inplace=True)
print(dataframe)


scaler = MinMaxScaler()
scaler.fit(dataframe[["petal length (cm)"]])
dataframe["petal length (cm)"] = scaler.transform(dataframe[["petal length (cm)"]])
scaler.fit(dataframe[["petal width (cm)"]])
dataframe["petal width (cm)"] = scaler.transform(dataframe[["petal width (cm)"]])


km = KMeans(n_clusters=3)
predicted = km.fit_predict(dataframe[["petal length (cm)","petal width (cm)"]])
print(predicted)

dataframe["cluster"] = predicted
print(dataframe)

cluster0 = dataframe[dataframe.cluster == 0]
cluster1 = dataframe[dataframe.cluster == 0]
cluster2 = dataframe[dataframe.cluster == 0]

print(cluster0)

# plt.ion()
# plt.xlabel("petal length (cm)")
# plt.ylabel("petal width (cm)")
# plt.scatter(cluster0["petal length (cm)"],cluster0["petal width (cm)"],marker="*",color ='red')
# plt.scatter(cluster1["petal length (cm)"],cluster1["petal width (cm)"],marker="*",color ='blue')
# plt.scatter(cluster2["petal length (cm)"],cluster2["petal width (cm)"],marker="*",color ='black')

print(dataset.target)

sse = []

for i in range(1,10):
    km = KMeans(i)
    predicted = km.fit_predict(dataframe[["petal length (cm)","petal width (cm)"]])
    sse.append(km.inertia_)


print(sse)

plt.xlabel("clusters")
plt.ylabel("sse")
plt.scatter(range(1,10),sse)