import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn


digit_set = load_digits()

dataframe = pd.DataFrame(digit_set.data)
print(dataframe)

plt.gray()
for i in range(5):
    plt.matshow(digit_set.images[i])


random_forest_model = RandomForestClassifier()

x_train,x_test,y_train,y_test = train_test_split(dataframe,digit_set.target,test_size=0.2)
print(len(x_train))

random_forest_model.fit(x_train,y_train)
print(random_forest_model.score(x_test,y_test))
predicted = random_forest_model.predict(x_test)

cm = confusion_matrix(y_test,predicted)
print(cm)
plt.figure(figsize=(10,7))
sbn.heatmap(cm,annot=True)
plt.xlabel("predicted")
plt.ylabel("Truth")