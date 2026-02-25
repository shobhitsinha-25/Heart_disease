import streamlit as st
import pandas as pd
import joblib 

model=joblib.load('svm_heart.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')

st.title('Heart Disease Prediction App')
st.markdown('This app predicts the likelihood of heart disease based on user input.')

# colect user input
age=st.number_input('Age', min_value=1, max_value=120, value=30)
sex=st.selectbox("Sex",['Male',"Female"])
resting_bp=st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
cholesterol=st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs=st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
rest_ecg=st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
max_heart_rate=st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exercise_angina=st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak=st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0)
slope=st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
Ca=st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, value=0)

if st.button('Predict'):
    input_data = {
    'Age': age,
    'Sex': 1 if sex == 'Male' else 0,
    'RestBP': resting_bp,
    'Chol': cholesterol,
    'Fbs': 1 if fbs == 'Yes' else 0,
    'RestECG': 0 if rest_ecg == 'Normal' else (1 if rest_ecg == 'ST-T wave abnormality' else 2),
    'MaxHR': max_heart_rate,
    'ExAng': 1 if exercise_angina == 'Yes' else 0,
    'Oldpeak': oldpeak,
    'Slope': 0 if slope == 'Upsloping' else (1 if slope == 'Flat' else 2),
    'Ca': Ca
}

    
    input_df=pd.DataFrame([input_data])
    
    scalered_input=scaler.transform(input_df[expected_columns])
    prediction=model.predict(scalered_input)

    if prediction[0]==1:
        st.error('The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.')
    else:
        st.success('The model predicts that you are unlikely to have heart disease. However, it is always a good idea to maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.')
    