import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn import model_selection

iris_set = load_iris()
dataframe = pd.DataFrame(iris_set.data,columns=iris_set.feature_names)
dataframe["flower"] = iris_set.target

reg = linear_model.LinearRegression()

reg.fit(dataframe[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]],iris_set.target)

print(reg.score(dataframe[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]],iris_set.target))

predicted = reg.predict([[5.1,3.5,1.4,0.2]]) 
print(predicted)
