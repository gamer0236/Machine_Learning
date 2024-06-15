import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


datframe= pd.read_csv("homeprices (1).csv")

le = LabelEncoder()
le_dataframe = datframe
le_dataframe.town = le.fit_transform(le_dataframe.town)                 
print(le_dataframe)

x_axis = le_dataframe[["town","area"]].values
y_axis= le_dataframe.price
print(x_axis)
print(y_axis)

ohe = OneHotEncoder(categories=[0])
X_axis = ohe.fit_transform(x_axis).tolist()
print(X_axis)