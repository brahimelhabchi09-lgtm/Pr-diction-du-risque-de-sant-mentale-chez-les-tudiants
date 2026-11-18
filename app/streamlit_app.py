import os
import requests
import streamlit as st


API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.title('Student Mental Health Prediction')

values = {
    'Age': st.number_input('Age', 17, 30, 21),
    'Year_of_Study': st.number_input('Year of study', 1, 5, 3),
    'Sleep_Hours': st.number_input('Sleep (hours/night)', 0., 24., 7.),
    'Study_Hours_Per_Week': st.number_input('Study (hours/week)', 0., 168., 20.),
    'GPA': st.number_input('GPA', 0., 4., 3.),
    'Social_Media_Hours': st.number_input('Social media (hours/day)', 0., 24., 4.),
    'Screen_Time_Hours': st.number_input('Screen time (hours/day)', 0., 24., 7.),
    'Financial_Stress': st.number_input('Financial stress', 1., 10., 5.),
    'Family_Support': st.number_input('Family support', 1., 10., 7.),
    'Attendance_Rate': st.number_input('Attendance (%)', 0., 100., 90.),
    'Gender': st.selectbox('Gender', ['Female', 'Male', 'Non-binary']),
    'Physical_Activity': st.selectbox('Physical activity', ['Low', 'Moderate', 'High']),
    'Part_Time_Job': st.selectbox('Part-time job', ['No', 'Yes']),
    'Diet_Quality': st.selectbox('Diet quality', ['Poor', 'Average', 'Good']),
    'Previous_Mental_Health_History': st.selectbox('Previous mental health history', ['No', 'Yes']),
}

if st.button('Predict'):
    try:
        response = requests.post(API_URL + '/predict', json=values, timeout=15)
        response.raise_for_status()
        st.write('Prediction:', response.json()['prediction'])
    except requests.RequestException:
        st.error('Start FastAPI and try again.')
