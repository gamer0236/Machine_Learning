import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt

dataframe = pd.read_csv("carprices (1).csv")
print(dataframe)

plt.ion()
plt.xlabel("age(yrs)")
plt.ylabel("price")
plt.scatter(dataframe["Age"],dataframe["SellPrice"],color = "red",marker=".")
# plt.plot(dataframe["Age"],dataframe["SellPrice"],color = "red")

x_axis = dataframe[["Mileage","Age"]]
y_axis = dataframe["SellPrice"]

# print(x_axis)
# print(y_axis)

x_train,x_test,y_train,y_test = model_selection.train_test_split(x_axis,y_axis,test_size=0.2)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
predicted = reg.predict(x_test)
print(predicted)
print(y_test)
print(reg.score(x_train,y_train))


