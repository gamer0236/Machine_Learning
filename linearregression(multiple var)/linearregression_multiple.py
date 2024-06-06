import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
# import word2number
# import word2number.w2n

dataframe = pd.read_csv("homeprices.csv")
print(dataframe)

medain = dataframe.bedrooms.median()
print(math.floor(medain))

dataframe = dataframe.fillna(medain)
print(dataframe)

plt.ion()
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(dataframe["area"],dataframe["price"],color = "red",marker= "*")
plt.plot(dataframe["area"],dataframe["price"],color = "blue")

reg = linear_model.LinearRegression()
reg.fit(dataframe[["area","bedrooms","age"]],dataframe.price)
prediction = reg.predict([[3000,4,30]])
print(math.floor(prediction))

# print(word2number.w2n.word_to_num("two million"))