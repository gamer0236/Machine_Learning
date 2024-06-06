import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# print(sklearn.__version__)

list = [[2600,550000],
        [3000,565000],
        [3200,610000],
        [3600,680000],
        [4000,725000]]

dataframe = pd.DataFrame(list,columns=["area","price"])
dataframe.to_csv("set.csv")
print(dataframe)

plt.ion()
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(dataframe.area,dataframe.price,color='red',marker='+')

reg = linear_model.LinearRegression()
reg.fit(dataframe[["area"]],dataframe.price)

# Predict the price for area = 3300
area_to_predict = np.array([[3300]])  # Reshape to 2D array
predicted_price = reg.predict(area_to_predict)
print(f"The predicted price for 3300 sq ft area is ${predicted_price[0]:.2f}")

# Plot the regression line
plt.plot(dataframe.area, reg.predict(dataframe[["area"]]), color='blue')