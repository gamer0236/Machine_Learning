import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler 

dataframe = pd.read_csv("income.csv")

print(dataframe)

# plt.ion()
# plt.xlabel("Age")
# plt.ylabel("Income")
# plt.scatter(dataframe.Age,dataframe.Income,marker="*",color="red")



scaler = MinMaxScaler()
scaler.fit(dataframe[["Income"]])
dataframe.Income = scaler.transform(dataframe[["Income"]])
scaler.fit(dataframe[["Age"]])
dataframe.Age = scaler.transform(dataframe[["Age"]])
print(dataframe)

km = KMeans(n_clusters=3)
precited = km.fit_predict(dataframe[["Age","Income"]])
print(precited)


dataframe["cluster"] = precited
print(dataframe)

cluster0 = dataframe[dataframe.cluster == 0]
cluster1 = dataframe[dataframe.cluster == 1]
cluster2 = dataframe[dataframe.cluster == 2]

plt.xlabel("Age")
plt.ylabel("Income")
plt.scatter(cluster0.Age,cluster0.Income,marker="*",color= "red")
plt.scatter(cluster1.Age,cluster1.Income,marker="*",color= "blue")
plt.scatter(cluster2.Age,cluster2.Income,marker="*",color= "black")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker=".",color = "purple")
plt.legend()

print(km.cluster_centers_)

sse = []

for i in range(1,10):
    km = KMeans(n_clusters=i)
    km.fit(dataframe[["Age","Income"]])
    sse.append(km.inertia_)


print(sse)

plt.xlabel("NoOfCluster")
plt.ylabel("SumOfSquaredError")
plt.plot(range(1,10),sse)