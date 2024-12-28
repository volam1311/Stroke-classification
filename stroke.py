import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data = pd.read_csv("stroke_classification.csv")
target = "stroke"
print(data.info())
"""
result = data.corr()
print(result)
"""
"""
sns.histplot(data["smokes"])
plt.title("Stroke")
plt.savefig("stroke.jpg")
"""
x = data.drop(target,axis =1)
x = data.drop("pat_id",axis =1)
y = data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 42)
num_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy = "median",fill_value = 0)),
    ("scaler",StandardScaler())
])
genders = ["Male","Female","Other"]
ordinal_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy = "most_frequent",fill_value = "unknown")),
    ("scaler",OrdinalEncoder(categories= [genders]))
])
preprocessor = ColumnTransformer(transformers =[
    ("num_features",num_transformer,["age","avg_glucose_level","bmi"]),
    ("ordinal_features",ordinal_transformer,["gender"]),
])
clf = Pipeline(steps = [
    ("preprocessor",preprocessor),
    ("classification",RandomForestClassifier())
])
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print(classification_report(y_test,y_predict))
