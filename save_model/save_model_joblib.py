import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

list = [[2600,550000],
        [3000,565000],
        [3200,610000],
        [3600,680000],
        [4000,725000]]

dataframe = pd.DataFrame(list,columns=["area","price"])

reg = linear_model.LinearRegression()
reg.fit(dataframe[["area"]],dataframe.price)
predicted = reg.predict([[5000]]) 
print(predicted)


# joblib import wen na