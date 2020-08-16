import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report

# feature studying
def study(data , feature) :
    survived = data[data["Survived"] == 1][feature].value_counts()
    dead = data[data["Survived"] == 0][feature].value_counts()
    study = pd.DataFrame([survived , dead])
    study.index = ["SURVIVED" , "DEAD"]
    study.plot(kind="bar")
    plt.show()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

allData = [train , test] # merging all data

# mapping features
sexMap = {"male": 0 , "female": 1}
embarkedMap = {"S": 0 , "C": 1 , "Q": 2}

# data cleaning
for data in allData :

    data["Sex"] = data["Sex"].map(sexMap)
    data["Embarked"] = data["Embarked"].map(embarkedMap)
    
    data["FamSize"] = data["SibSp"] + data["Parch"] + 1
    data.drop(["SibSp" , "Parch"] , axis=1 , inplace=True)

    data["Age"].fillna(data.groupby("Sex")["Age"].transform("mean") , inplace=True)
    data.loc[ (data["Age"] <= 18) , "Age" ] = 0
    data.loc[ (data["Age"] > 18) & (data["Age"] <= 30) , "Age" ] = 1
    data.loc[ (data["Age"] > 30) & (data["Age"] <= 55) , "Age" ] = 2
    data.loc[ (data["Age"] > 55) , "Age" ] = 3

    data.loc[ (data["Fare"] <= 8) , "Fare" ] = 0
    data.loc[ (data["Fare"] > 8) & (data["Fare"] <= 16) , "Fare" ] = 1
    data.loc[ (data["Fare"] > 16) & (data["Age"] <= 35) , "Fare" ] = 2
    data.loc[ (data["Fare"] > 35) , "Fare" ] = 3

    data.drop(["Name" , "Ticket" , "Cabin"] , axis=1 , inplace=True)

# print(allData[0].tail(6))
# print(allData[0]["Fare"].describe())
# print(allData[0].isnull().sum())
# study(allData[0] , "Fare")

x = allData[0].drop(["PassengerId" , "Survived"] , axis=1)
y = allData[0]["Survived"]

x = np.nan_to_num(x)
y = np.nan_to_num(y)

x_train , x_test , y_train , y_test = model_selection.train_test_split(x , y)

# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(max_depth=5)
clf.fit(x_train , y_train)

y_test_pred = clf.predict(x_test)
print(confusion_matrix(y_test , y_test_pred))
print(clf.score(x_train , y_train) , clf.score(x_test , y_test))

x_ques = allData[1].drop(["PassengerId"] , axis=1)
x_ques = np.nan_to_num(x_ques)

y_ans = clf.predict(x_ques)

soln = pd.DataFrame()
soln["PassengerId"] = allData[1]["PassengerId"]
soln["Survived"] = y_ans

print(soln)
soln.to_csv("soln.csv" , index=False)