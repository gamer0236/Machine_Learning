import pandas as pd
from sklearn import linear_model
import numpy as np

dataframe = pd.read_csv("carprices.csv")
print(dataframe)

dummy_dataframe = pd.get_dummies(dataframe.CarModel,dtype=int)
dataframe.drop(["CarModel"],axis=1,inplace=True)
merged_dataframe = pd.concat([dummy_dataframe,dataframe],axis="columns") 
merged_dataframe.drop(["Mercedez Benz C class"],axis=1,inplace=True)
print(merged_dataframe)

x_axis = merged_dataframe.drop(["SellPrice"], axis="columns")
print(x_axis)
y_axis = dataframe.SellPrice
print(y_axis)

reg = linear_model.LinearRegression()
reg.fit(x_axis,y_axis)
predicted = reg.predict([[0,1,70000,7]])
print(predicted)
print(reg.score(x_axis,y_axis))