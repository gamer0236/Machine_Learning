import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB


dataframe = pd.read_csv("titanic.csv")
print(dataframe)
print(dataframe.columns)
target = dataframe["Survived"]
dataframe.drop(["Survived","PassengerId","SibSp","Cabin","Ticket","Embarked","Name","Parch"],axis="columns",inplace=True)
print(dataframe)
print(target)

encoder = LabelEncoder()

# dataframe.Sex = encoder.fit_transform(dataframe["Sex"])
dummy_df = pd.get_dummies(dataframe["Sex"],dtype=int) 
print(dummy_df)

dataframe.drop(["Sex"],axis="columns",inplace=True)

dataframe = pd.concat([dataframe,dummy_df],axis="columns")
print(dataframe)

print(dataframe.columns[dataframe.isna().any()])

replacement_no = dataframe["Age"].mean()
print(replacement_no)
dataframe.Age = dataframe.Age.fillna(replacement_no)
dataframe.Age = dataframe["Age"].round()
print(dataframe["Age"])
print(dataframe)

x_train,x_test,y_train,y_test = train_test_split(dataframe,target,test_size=0.2)
print(len(x_train))

gaussian_model = GaussianNB()

gaussian_model.fit(x_train,y_train)
print(gaussian_model.score(x_test,y_test))


