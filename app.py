import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler as ss

# Loading our model
loaded_model = pickle.load(open('D:/DGM WEB APP/logistic_regression_model_evaluation.pkl','rb'))

# Custom inputs to check whether our loaded model will predict it or not.
input= (1.28,0.28,1.13,0.35,1,6,0,0)

# Changing the input data to numpy array and reshaping it as we are predicting one instance
input_array=np.asarray(input).reshape(1,-1)

# Predicting the output from loaded model
prediction = loaded_model['model'].predict(input_array)
if (prediction==0):
    print("Person will not click on ad")
else:
    print("Person will click on ad")

#-------------------------Till now we have seen our loaded model is working fine, NOw lets start creatin a web app with scaling some user inputs to maintain consistency of data-------------------------------------------#
    
            