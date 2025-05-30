import streamlit as st
import pandas as pd
import pickle

# Load model dan scaler
with open('/content/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/content/minmaxscaler.pkl', 'rb') as f:
    minmaxscaler = pickle.load(f)

st.title("Student Dropout Prediction App 🎓")
st.write("Masukkan data mahasiswa untuk memprediksi apakah akan dropout atau tidak.")

# Form input
with st.form("dropout_form"):
    col1, col2 = st.columns(2)

    with col1:
        cu2_approved = st.number_input("2nd Sem Approved Units", min_value=0)
        cu2_grade = st.number_input("2nd Sem Grade", min_value=0.0)
        cu1_approved = st.number_input("1st Sem Approved Units", min_value=0)
        tuition_paid = st.selectbox("Tuition Fees Up to Date", [1, 0])
        cu1_grade = st.number_input("1st Sem Grade", min_value=0.0)

    with col2:
        age = st.number_input("Age at Enrollment", min_value=16)
        admission_grade = st.number_input("Admission Grade", min_value=0.0)
        prev_grade = st.number_input("Previous Qualification Grade", min_value=0.0)
        cu2_evals = st.number_input("2nd Sem Evaluations", min_value=0)
        course = st.number_input("Course (encoded)", min_value=0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade,
            'Curricular_units_1st_sem_approved': cu1_approved,
            'Tuition_fees_up_to_date': tuition_paid,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Age_at_enrollment': age,
            'Admission_grade': admission_grade,
            'Previous_qualification_grade': prev_grade,
            'Curricular_units_2nd_sem_evaluations': cu2_evals,
            'Course': course
        }

        df = pd.DataFrame([input_data])
        df_scaled = pd.DataFrame(minmaxscaler.transform(df), columns=df.columns)

        prediction = model.predict(df_scaled)[0]
        result = 'Dropout' if prediction == 1 else 'Not Dropout'

        st.success(f"Prediction: **{result}**")
