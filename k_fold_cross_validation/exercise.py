import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dataset = load_iris()

logistic_reg = LogisticRegression()
random_forest = RandomForestClassifier()
support_vector = SVC()

fold = KFold()

print(cross_val_score(logistic_reg,dataset.data,dataset.target,cv=3))
print(cross_val_score(random_forest,dataset.data,dataset.target,cv=3))
print(cross_val_score(support_vector,dataset.data,dataset.target,cv=3))
