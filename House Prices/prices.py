import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix , classification_report

# feature studying
def study(data , feature) :
    expensive = data[data["SalePrice"] >= 163000][feature].value_counts()
    cheap = data[data["SalePrice"] < 163000][feature].value_counts()
    study = pd.DataFrame([expensive , cheap])
    study.index = ["exp" , "chp"]
    study.plot(kind="bar")
    plt.show()

def make_map(data) :
    mapped = {}
    keys = set(data)
    value = 0
    for title in keys :
        mapped[title] = value
        value += 1
    return mapped

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

allData = [train , test] # merging all data

# print(train.SalePrice.describe())

# print(train["GarageYrBlt"].describe())
# print(train["GarageYrBlt"].value_counts())
# print(np.isnan(train.Street).sum())
# study(train , "PoolArea")

mapping_cols = ['MSSubClass' , 'MSZoning' , 
        'LotArea' , 'LotShape' , 'LandContour' , 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'BldgType',
       'HouseStyle' , 'OverallQual', 'OverallCond',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC',
       'Fence', 'MiscFeature', 'MoSold', 'SaleType',
       'SaleCondition']

maps = {}

for curCol in mapping_cols :
    maps[curCol] = make_map(train[curCol])

for data in allData :
    for title in maps :
        data[title] = data[title].map(maps[title])
    
    data["LotArea"].fillna(data["LotArea"].mean() , inplace=True)
    data["LotFrontage"].fillna(data["LotFrontage"].mean() , inplace=True)
    data["MasVnrArea"].fillna(data["MasVnrArea"].mean() , inplace=True)
    data["GarageYrBlt"].fillna(int(data["GarageYrBlt"].mode()) , inplace=True)

    data["EffectiveBsmtSF"] = data["TotalBsmtSF"] - data["BsmtUnfSF"]
    data["TotalBathrooms"] = data["BsmtFullBath"] + data["FullBath"] + 0.5 * (data["BsmtHalfBath"] + data["HalfBath"])
    data["PorchArea"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
    
    data.drop(["Utilities" , "Street" , "Alley" , "Condition2" , "YearBuilt" , "BsmtUnfSF" , "TotalBsmtSF" , "FullBath" , "HalfBath" , "BsmtFullBath" , "BsmtHalfBath" , "Fireplaces" , 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'ScreenPorch' , "PoolArea" , "MiscVal"] , axis=1 , inplace=True)

x = allData[0].drop(["Id" , "SalePrice"] , axis=1)
y = allData[0]["SalePrice"]

### trial and error

# for i in x.columns :
#     print(i , np.isnan(x[i]).sum())
# print()

# x_train , x_test , y_train , y_test = model_selection.train_test_split(x , y , random_state = 987)
# scaler = preprocessing.StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)

# slr = RandomForestRegressor(max_features="sqrt" , max_depth=5 , random_state=100)
# slr.fit(x_train , y_train)

# x_test = scaler.transform(x_test)
# print(slr.score(x_train , y_train))
# print(slr.score(x_test , y_test))

### KAGGLE SUBMISSION

scaler = preprocessing.StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

slr = RandomForestRegressor(max_features="log2")
slr.fit(x , y)

x_test = allData[1].drop(["Id"] , axis=1)

# for i in x_test.columns :
#     print(i , np.isnan(x_test[i]).sum())

x_test = np.nan_to_num(x_test)

x_test = scaler.transform(x_test)

y_pred = slr.predict(x_test)
print(y_pred)

soln = pd.DataFrame()
soln["Id"] = allData[1]["Id"]
soln["SalePrice"] = y_pred

print(soln)
soln.to_csv("soln.csv" , index=False)