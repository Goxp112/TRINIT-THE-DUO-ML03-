import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle
from flask import Flask,render_template,request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings(action = 'ignore')
app=Flask(__name__,template_folder='html')
@app.route("/")
def home():
    data=pd.read_csv(r'C:\Users\niles\OneDrive\Desktop\crop prediction/final crop dataset.csv')
    label_encoder = preprocessing.LabelEncoder()
    data['soil']= label_encoder.fit_transform(data['soil'])
    data['soil'].unique()
    X=data[['N', 'P', 'K', 'temperature', 'humidity', 'ph',
    'rainfall', 'min_price', 'max_price', 'modal_price',
    'soil']]
    y=data[['commodity']]
    smote = SMOTE()
    X_train_os,y_train_os=smote.fit_resample(np.array(X),np.array(y))
    X_train, X_test, y_train, y_test = train_test_split(X_train_os, y_train_os,test_size = 0.2, random_state = 202)
    rf_model=RandomForestClassifier(random_state=200000,max_depth=2,n_estimators=300,min_samples_leaf=7)
    rf_model.fit(X_train,y_train)
    pickle.dump(rf_model,open("nilesh.pkl","wb"))
    return render_template("home.html")
@app.route("/predict",methods=["GET","POST"])

def predict():
    N=request.form['N']
    P=request.form['P']
    K=request.form['K']
    temperature=request.form['temperature']
    humidity=request.form['humidity']
    ph=request.form['ph']
    rainfall=request.form['rainfall']
    min_price=request.form['min_price']
    max_price=request.form['max_price']
    modal_price=request.form['modal_price']
    soil=request.form['soil']
    form_array=np.array([[N,P,K,temperature,humidity,ph,rainfall,min_price,max_price,modal_price,soil]])
    rf_model=pickle.load(open("nilesh.pkl","rb"))
    prediction=rf_model.predict(form_array)[0]
    if prediction==11:
        result="rice"
    elif prediction==6:
        result="maize"
    elif prediction==10:
        result="pomegranate"
    elif prediction==1:
        result="banana"
    elif prediction==7:
        result="mango"
    elif prediction==4:
        result="grapes"
    elif prediction==0:
        result="apple"
    elif prediction==8:
        result="orange"
    elif prediction==9:
        result="papaya"
    elif prediction==2:
        result="coconut"
    elif prediction==3:
        result="cotton"
    elif prediction==5:
        result="jute"
    else:
        result = "no crop can be grown"
    return render_template("result.html", result=result)
if __name__ == '__main__':  
   app.run()