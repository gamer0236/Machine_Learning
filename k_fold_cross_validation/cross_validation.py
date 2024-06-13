import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier 

dataset = load_digits()
print(dir(dataset))
dataframe = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(dataframe)

stratified_kfold = StratifiedKFold(n_splits=3)
fold = KFold(n_splits=3)

# for train_set,test_set in fold.split([1,2,3,4,5,6,7,8,9]):
#     print(train_set,end=" ")
#     print(test_set)
logistic_fun = LogisticRegression()
forest = RandomForestClassifier()
# listk = [1,2,3,4,5,6,7,8,9]
logistic_fun_list = []
forest_list = []

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

for train_index,test_index in fold.split(dataset.data):
    x_train,x_test,y_train,y_test = dataset.data[train_index],dataset.data[test_index],dataset.target[train_index],dataset.target[test_index]
    # print(get_score(forest,x_train,x_test,y_train,y_test))
    forest.fit(x_train,y_train)
    logistic_fun_list.append(forest.score(x_test,y_test))

    logistic_fun.fit(x_train,y_train)
    forest_list.append(logistic_fun.score(x_test, y_test))

print(logistic_fun_list)
print(forest_list)


#oya for loop ek nathuwa eka wenuwat kelinma use krnna plwn cross_val function ek
# dekenma wenne ekama de

print(cross_val_score(forest, dataset.data, dataset.target, cv=3))
print(cross_val_score(logistic_fun, dataset.data, dataset.target, cv=3)) 
