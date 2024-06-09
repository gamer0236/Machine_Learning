import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import seaborn as seb

irish_dataset = load_iris()
print(dir(irish_dataset)) 
print(irish_dataset.target[1])

logistic_reg = linear_model.LogisticRegression()



x_train,x_test,y_train,y_test = model_selection.train_test_split(irish_dataset.data,irish_dataset.target,test_size=0.2)

logistic_reg.fit(x_train,y_train)
print(logistic_reg.score(x_train,y_train))


predicted = logistic_reg.predict(x_test)
print(predicted)

print(irish_dataset.feature_names[0])

cm = confusion_matrix(y_test,predicted)
print(cm)
plt.figure(figsize=(3,3))
seb.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("Truth")