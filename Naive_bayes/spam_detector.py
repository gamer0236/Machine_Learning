import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv("spam.csv")
dataset["spam"] = dataset["Category"].apply(lambda x: 1 if x == "spam" else 0)
print(dataset)

x_train,x_test,y_train,y_test = train_test_split(dataset["Message"],dataset["spam"],test_size=0.25)

vector = CountVectorizer()

x_train_converted = vector.fit_transform(x_train.values)

x_train_converted.toarray()[:3]
print(x_train_converted)


model = MultinomialNB()

model.fit(x_train_converted,y_train)


emails = ["Het mohan,can we get together to watch footbal game tomorrow?",
          "upto 20 discount on parking,exclusive offer just for you.dont miss this reward!"]

emails_count = vector.transform(emails)
print(model.predict(emails_count))

x_test_count = vector.transform(x_test)
print(model.score(x_test_count,y_test))


# pipeline


clf = Pipeline([
    ("vectorizer",CountVectorizer()),
    ("nb",MultinomialNB())
])

clf.fit(x_train,y_train)
print(clf.predict(emails))
