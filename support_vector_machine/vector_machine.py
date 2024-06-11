import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import model_selection
from sklearn import svm

iris_set = load_iris()

dataframe = pd.DataFrame(iris_set.data,columns=iris_set.feature_names)
print(dataframe)
dataframe["target"] = iris_set.target
print(dataframe)
dataframe["flower_name"] = dataframe.target.apply(lambda x : iris_set.target_names[x])
print(dataframe)

setosa_df = dataframe[dataframe.target == 0]
versicolor_df = dataframe[dataframe.target == 1]
virginica_df = dataframe[dataframe.target == 2]
print(setosa_df)

plt.ion()
plt.xlabel("sepal lenght (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(setosa_df["sepal length (cm)"],setosa_df["sepal width (cm)"],marker="*",color = 'red')
plt.scatter(versicolor_df["sepal length (cm)"],versicolor_df["sepal width (cm)"],marker="*",color = 'blue')

# plt.xlabel("petal lenght (cm)")
# plt.ylabel("petal width (cm)")
# plt.scatter(setosa_df["petal length (cm)"],setosa_df["petal width (cm)"],marker="*",color = 'red')
# plt.scatter(versicolor_df["petal length (cm)"],versicolor_df["petal width (cm)"],marker="*",color = 'blue')

# print(dataframe)
x_axis = dataframe.drop(["target","flower_name"],axis='columns')
print(x_axis) 

y_axis = dataframe["target"]
print(y_axis)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_axis, y_axis, test_size=0.2)
print(len(x_train), len(x_test), len(y_train), len(y_test))

model = svm.SVC()

model.fit(x_train,y_train)
print(model.score(x_train,y_train))

predicted_array = model.predict(x_test)

mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

converted_array = [mapping[i] for i in predicted_array]

print(converted_array) 