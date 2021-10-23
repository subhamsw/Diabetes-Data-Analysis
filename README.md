# Diabetes-Data-Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data =  pd.read_csv("/diabetes_type_dataset.csv")
data.dtypes

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["DIABETES_TYPE"])
data["DIABETES_TYPE"] = enc.transform(data["DIABETES_TYPE"])

data["SYNC_COUNT"] = data["SYNC_COUNT"].astype(object)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["SYNC_COUNT"])
data["SYNC_COUNT"] = enc.transform(data["SYNC_COUNT"])

data["EVENT_COUNT"] = data["EVENT_COUNT"].astype(object)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["EVENT_COUNT"])
data["EVENT_COUNT"] = enc.transform(data["EVENT_COUNT"])

data["FOOD_COUNT"] = data["FOOD_COUNT"].astype(object)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["FOOD_COUNT"])
data["FOOD_COUNT"] = enc.transform(data["FOOD_COUNT"])

data["EXERCISE_COUNT"] = data["EXERCISE_COUNT"].astype(object)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["EXERCISE_COUNT"])
data["EXERCISE_COUNT"] = enc.transform(data["EXERCISE_COUNT"])

data["MEDICATION_COUNT"] = data["MEDICATION_COUNT"].astype(object)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["MEDICATION_COUNT"])
data["MEDICATION_COUNT"] = enc.transform(data["MEDICATION_COUNT"])

data.tail()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(data["GENDER"])
data["GENDER"] = enc.transform(data["GENDER"])

data.tail(100)

data = data.dropna()

data.dtypes

#Note: Use scatterplot to check the diagram curve
#Next Step:
data.columns

feature_data = data[["GENDER", "CURRENT_AGE", "AGE_WHEN_REGISTERED",
       "SYNC_COUNT", "EVENT_COUNT", "FOOD_COUNT", "EXERCISE_COUNT", "MEDICATION_COUNT"]]

#Independent var
X = np.asarray(feature_data)

# Dependent var
y = np.asarray(data["DIABETES_TYPE"])

y[0:20]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train.shape

y_train.shape

X_test.shape

y_test.shape

from sklearn import svm

classifier = svm.SVC(kernel="linear", gamma="auto", C=2)

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))


I have used support vector machine but also possible to use other classification algorithms. There is possible to use random forest also.
