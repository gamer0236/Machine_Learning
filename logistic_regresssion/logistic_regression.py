import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model

dataframe = pd.read_csv("insurance_data.csv")
print(dataframe)

x_axis = dataframe[["age"]]
y_axis = dataframe["bought_insurance"]
print(x_axis)
print(y_axis)

plt.ion()
plt.xlabel("age")
plt.ylabel("bought_insurance")
plt.scatter(dataframe["age"],dataframe["bought_insurance"],color = "red",marker="*")

x_train,x_test,y_train,y_test = model_selection.train_test_split(x_axis,y_axis,test_size=0.1)

LogisticReg = linear_model.LogisticRegression()
LogisticReg.fit(x_train,y_train)
print(x_test)
test_predicted = LogisticReg.predict(x_test)
print(test_predicted)  
print(LogisticReg.score(x_test,y_test))