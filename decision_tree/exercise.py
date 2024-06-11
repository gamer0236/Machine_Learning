import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

titanic_datframe = pd.read_csv("titanic.csv")
print(titanic_datframe)

x_axis = titanic_datframe.drop(columns=["Survived","PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"])
y_axis =titanic_datframe["Survived"]
print(x_axis)

encoder = LabelEncoder()

x_axis["Sex"] = encoder.fit_transform(titanic_datframe["Sex"])
print(x_axis)
print(y_axis)

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_axis,y_axis)
print(tree_model.score(x_axis,y_axis))
predicted = tree_model.predict([[1,0,38,70]])
print(predicted)