import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

list = [[2600,550000],
        [3000,565000],
        [3200,610000],
        [3600,680000],
        [4000,725000]]

dataframe = pd.DataFrame(list,columns=["area","price"])

reg = linear_model.LinearRegression()
reg.fit(dataframe[["area"]],dataframe.price) 

with open("model_pickle",'wb') as file:
    pickle.dump(reg,file)

with open("model_pickle",'rb') as file:
    model = pickle.load(file)
    
model.predict([[6000]])