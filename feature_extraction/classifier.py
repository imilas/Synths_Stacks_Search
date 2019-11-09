#load the data
import numpy as np
import pandas as pd
import random
import mir_utils as miru
import sounddevice as sd
import matplotlib.pyplot as plt

testFraction=0.9

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df=pd.read_csv("feat_frequency_bins.csv")
#preperocessing
le = preprocessing.LabelEncoder()
le.fit(df.label)
le.transform(df.label)
df.label=le.transform(df.label)
print(np.sum(df.dtypes!=np.float64)) #make sure this is 1, our label column is string
#df.label=le.inverse_transform(df.label) #inverse of transform if u need it

X=df.loc[:,df.columns!="label"]
y=df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFraction, random_state=42)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',verbose=True)
svclassifier.fit(X_train, y_train,)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))