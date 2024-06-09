import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn.metrics import confusion_matrix


digits_dataset = load_digits()

plt.gray()
plt.matshow(digits_dataset.images[4])
print(digits_dataset.data[4])

x_train,x_test,y_train,y_test = model_selection.train_test_split(digits_dataset.data,digits_dataset.target,test_size= 0.2)
print(len(x_train))

logistic_reg = linear_model.LogisticRegression()
logistic_reg.fit(x_train,y_train)
print(logistic_reg.score(x_train,y_train))
predicted =  logistic_reg.predict([digits_dataset.data[67]])
print(predicted)

y_precited = logistic_reg.predict(x_test)

cm = confusion_matrix(y_test,y_precited)
print(cm)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')