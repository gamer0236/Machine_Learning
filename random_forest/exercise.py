import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix

iris_datset = load_iris()

dataframe = pd.DataFrame(iris_datset.data,columns=iris_datset.feature_names)
dataframe["target"] = iris_datset.target
print(dataframe)

x_axis = dataframe.drop(["target"],axis='columns')
print(x_axis)

y_axis = dataframe["target"]
print(y_axis) 

random_tree_model = RandomForestClassifier(n_estimators=100)

x_train,x_test,y_train,y_test = train_test_split(x_axis,y_axis,test_size=0.2)

random_tree_model.fit(x_train,y_train)
print(random_tree_model.score(x_train,y_train))
print(len(x_test))

predicted = random_tree_model.predict(x_test)
print(predicted)

cm = confusion_matrix(y_test,predicted)
print(cm)
plt.figure(figsize=(10,7))
sbn.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("Truth")