import pandas as pd
import numpy as np
import math
from word2number import w2n
from sklearn import linear_model

dataframe = pd.read_csv("hiring.csv")
print(dataframe)

dataframe.experience = dataframe.experience.fillna("zero")
print(dataframe)

dataframe.experience = dataframe.experience.apply(w2n.word_to_num)
# alternative

# for element in range(len(dataframe)):
#     dataframe["experience"][element] = w2n.word_to_num(dataframe["experience"][element])   
print(dataframe)

median = dataframe.test_score.median()
dataframe = dataframe.fillna(median)
print(dataframe)

reg = linear_model.LinearRegression()
reg.fit(dataframe[["experience","test_score","interview_score(out of 10)"]],dataframe.salary)
prediction = reg.predict([[2,4,5]])
print(math.floor(prediction))



