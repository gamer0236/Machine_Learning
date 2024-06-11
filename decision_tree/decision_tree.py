import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

dataframe = pd.read_csv("salaries.csv")
print(dataframe)

x_axis = dataframe.drop(columns=["salary_more_then_100k"],axis="columns")
print(x_axis)
y_axis = dataframe["salary_more_then_100k"]
print(y_axis)

encoder = LabelEncoder()

x_axis["company"] = encoder.fit_transform(dataframe["company"])
x_axis["job"] = encoder.fit_transform(dataframe["job"])
x_axis["degree"] = encoder.fit_transform(dataframe["degree"])
print(x_axis)

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_axis,y_axis)
print(tree_model.score(x_axis,y_axis))
predicted = tree_model.predict([[2,2,0]])
print(predicted)
