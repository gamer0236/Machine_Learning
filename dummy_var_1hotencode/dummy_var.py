import pandas as pd
from sklearn import linear_model


dataframe = pd.read_csv("homeprices (1).csv")
print(dataframe)

dummy_df  = pd.get_dummies(dataframe.town,dtype=int)
print(dummy_df)

dataframe.drop(["town"], axis=1, inplace=True)
merged_dataframe = pd.concat([dataframe,dummy_df],axis = "columns")
merged_dataframe.drop(["west windsor"], axis=1, inplace=True)
print(merged_dataframe)

reg = linear_model.LinearRegression()
x_axis = merged_dataframe.drop(["price"], axis="columns")
y_axis = dataframe["price"]
print(x_axis)
print(y_axis)
reg.fit(x_axis,y_axis)
predicted = reg.predict([[2800,0,1]])
print(predicted)
print(reg.score(x_axis,y_axis))

