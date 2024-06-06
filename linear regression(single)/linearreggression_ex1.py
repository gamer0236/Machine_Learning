import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn import linear_model;

dataframe = pd.read_csv("canada_per_capita_income.csv")
dataframe.columns = ["year","percapitaincome"]
print(dataframe)

plt.ion()
plt.xlabel("year")
plt.ylabel("income")
plt.scatter(dataframe.year,dataframe.percapitaincome,color='red',marker='+')
plt.plot(dataframe.year,dataframe.percapitaincome,color='blue')

reg = linear_model.LinearRegression()
reg.fit(dataframe[["year"]],dataframe.percapitaincome)

year_to_predict = np.array([[2020]])
predicted_income = reg.predict(year_to_predict)

print(f"predicted income of 2020 is {predicted_income[0]:.2f}$")
plt.plot(dataframe.year,dataframe.percapitaincome,color='blue')
