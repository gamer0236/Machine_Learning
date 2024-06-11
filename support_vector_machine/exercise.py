import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import model_selection


digit_set = load_digits()
print(dir(digit_set))

model = svm.SVC(C=100, gamma=0.1)
x_train,x_test,y_train,y_test = model_selection.train_test_split(digit_set.data,digit_set.target,test_size=0.2)

model.fit(x_train,y_train)
print(model.score(x_train,y_train)) 
