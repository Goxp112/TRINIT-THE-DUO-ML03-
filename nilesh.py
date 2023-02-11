# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:24:36 2022

@author: hp
"""

import numpy as np
import pandas as pd
import pickle
from flask import Flask,render_template,request
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
import warnings
from termcolor import colored
warnings.filterwarnings(action = 'ignore')
app=Flask(__name__,template_folder='html')
@app.route("/")
def home():
    data=pd.read_csv(r'C:\Users\hp\OneDrive\Documents/breastcancer.csv')
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data= data.fillna(0)
    X = data[['radius_mean','texture_mean','perimeter_mean','area_mean']]
    y = data['diagnosis']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=18)
    rf_model=RandomForestClassifier(random_state=50,n_estimators=10,max_depth=6,min_samples_leaf=5)
    rf_model.fit(X_train,y_train)
    pickle.dump(rf_model,open("nilesh.pkl","wb"))
    return render_template("home.html")
@app.route("/predict",methods=["GET","POST"])

def predict():
    radius_mean=request.form['radius_mean']
    texture_mean=request.form['texture_mean']
    perimeter_mean=request.form['perimeter_mean']
    area_mean=request.form['area_mean']
    form_array=np.array([[radius_mean,texture_mean,perimeter_mean,area_mean]])
    rf_model=pickle.load(open("nilesh.pkl","rb"))
    prediction=rf_model.predict(form_array)[0]
    if prediction==0:
        result="M"
    else:
        result="B"
    return render_template("result.html", result=result)

if __name__=="__main__":
    app.run(debug=False)
    