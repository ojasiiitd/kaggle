import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

x_train = train.drop("label" , axis=1)
y_train = train["label"]

# print(x_train.shape)
# print(y_train.shape)

pca = PCA()
pca.fit_transform(x_train)
total_var = sum(pca.explained_variance_)
pixels = 0
cur_var = 0
while (cur_var/total_var) < 0.95 :
    cur_var += pca.explained_variance_[pixels]
    pixels += 1
print("USEFUL PIXELS:" , pixels)

pca = PCA(n_components=pixels)
x_train_reduced = pca.fit_transform(x_train)

clf = SVC(max_iter=1000)
clf.fit(x_train_reduced , y_train)

x_test = test
x_test_reduced = pca.transform(x_test)
img_id = np.arange(1 , len(x_test_reduced)+1 , 1)
predicted_labels = clf.predict(x_test_reduced)

soln = pd.DataFrame()
soln["ImageId"] = img_id
soln["Label"] = predicted_labels

soln.to_csv("soln.csv" , index=False)