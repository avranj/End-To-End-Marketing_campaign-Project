import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler as ss

# load the model
load_model= pickle.load(open('D:/DGM WEB APP/logistic_regression_model_evaluation.pkl','rb'))


# We are now saling the some inputs from user.

def scale_the_user_data(user_inputs):

     # Filter out non-numerical inputs
    numerical_inputs = [x for x in user_inputs if isinstance(x, (int, float)) and x not in [0, 1]]
    
    if not numerical_inputs:
        print("No numerical inputs found.")
        return user_inputs

    sc = ss()
    X = np.array(numerical_inputs).reshape(-1, len(numerical_inputs))
    scaled_cols = sc.fit_transform(X)

    scaled_data = user_inputs.copy()
    for i, col_index in enumerate([i for i, val in enumerate(user_inputs) if val in numerical_inputs]):
        scaled_data[col_index] = scaled_cols[:, i]

    print(scaled_data)

    return scaled_data


# We are creating a function for prediction 
def Advertisement_prediction(userInput):
    onedarray=np.asarray(userInput).reshape(1,-1)
    predicted_value=load_model['model'].predict(onedarray)
    print("Predicted Value is :- ",predicted_value)

    if predicted_value==1:
        return "User will click on Advertisement"
    else:
        return "User will not click on the Advertisement"
    
# From here we will now start creating our web app

def main():

    st.title('Driving Sales Through Effective Conversions')

    # ['Daily Internet Usage', 'Daily Time Spent on Site', 'Age', 'Area Income','hour_18-24', 'weekday', 'hour_7-12', 'hour_1-6']

    Daily_Internet_Usage= st.number_input('Daily Internet Usage',step=1)

    Daily_Time_spent_onsite= st.number_input('Daily Time Spnt On Site',step=1)

    Age = st.number_input('User Age',step=1)

    Area_income = st.number_input('Income of User',step=1)
    # Taking input for late night as a radio button
    Active_late_night = st.radio("Are you active on very late night on site",
                                 ['Active-1','Not-Active-0'],key='radio1')
    active_late_night_options={'Active-1':1,'Not-Active-0':0}

    Active_late_night_encoded= active_late_night_options[Active_late_night]
    # Taking a input for a weekday as a radio button
    day_of_week = st.radio(
        "What is the day today you are using site",
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],key='radio2'
    ) 
    weekday_encoded = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

    weekday = weekday_encoded[day_of_week]
    
    # Taking input for active in morning slot
    Active_early_morning = st.radio("Are you active on very early morning on site",
                                 ['Active-1','Not-Active-0'],key='radio3')
    active_early_morning_options={'Active-1':1,'Not-Active-0':0}

    Active_early_morning_encoded= active_early_morning_options[Active_early_morning]

    # Taking input for active in Afternoon slot

    Active_afternoon = st.radio("Are you active on very early morning on site",
                                 ['Active-1','Not-Active-0'],key='radio4')
    active_afternoon_options={'Active-1':1,'Not-Active-0':0}

    Active_afternoon_encoded= active_afternoon_options[Active_afternoon]

    # Now we have gathered all the user input data, Let's standardize some values to maintain consistency of data

    user_inputs= [Daily_Internet_Usage, Daily_Time_spent_onsite, Age, Area_income, Active_late_night_encoded, 
                  weekday, Active_early_morning_encoded, Active_afternoon_encoded]
    
    scaled_data = scale_the_user_data(user_inputs)

    # Now we will give this scaled data to predictive function using button and predict the result

    result=''

    if st.button('Marketing Campaign click Interest result'):
        result=Advertisement_prediction(scaled_data)

    st.success(result)


if __name__== '__main__':
    main()        






    
