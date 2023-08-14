import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pickle



data_train = pd.read_csv("E:\PROJECTS\LOAN APPROVAL PREDICTION\Training Data.csv")
data_train_cleaned = data_train.drop(columns=["Id"])
data_train_cleaned["Risk_Flag"] = data_train_cleaned["Risk_Flag"].map({1: "YES", 0: "NO"})

data_train_con = data_train_cleaned[["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]]
data_train_cat = data_train_cleaned[["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]]

cols = ['Married/Single', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag', 'House_Ownership']
le = preprocessing.LabelEncoder()
for col in cols:
    data_train_cleaned[col] = le.fit_transform(data_train_cleaned[col])

df_tmp = data_train_cleaned
y = df_tmp['Risk_Flag']
X = df_tmp.drop(columns=['Risk_Flag'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X_train, y_train)

pickle.dump(model,open("model.pkl","wb"))