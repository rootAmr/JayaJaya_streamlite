import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi halaman
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
st.title("🎓 Student Dropout Prediction App")
st.write("Masukkan data mahasiswa untuk memprediksi kemungkinan dropout.")

# Path ke file model
model_path = 'model.joblib'

# Validasi keberadaan file
if not os.path.exists(model_path):
    st.error("Model tidak ditemukan. Pastikan file 'model.joblib' tersedia.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Form input user
with st.form("dropout_form"):
    st.subheader("🔢 Masukkan Data Mahasiswa:")
    col1, col2 = st.columns(2)

    with col1:
        cu2_approved = st.number_input("2nd Sem Approved Units", min_value=0, step=1)
        cu2_grade = st.number_input("2nd Sem Grade", min_value=0.0, step=0.1)
        cu1_approved = st.number_input("1st Sem Approved Units", min_value=0, step=1)
        tuition_paid = st.selectbox("Tuition Fees Up to Date", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        cu1_grade = st.number_input("1st Sem Grade", min_value=0.0, step=0.1)

    with col2:
        age = st.number_input("Age at Enrollment", min_value=16, step=1)
        admission_grade = st.number_input("Admission Grade", min_value=0.0, step=0.1)
        prev_grade = st.number_input("Previous Qualification Grade", min_value=0.0, step=0.1)
        cu2_evals = st.number_input("2nd Sem Evaluations", min_value=0, step=1)
        course = st.number_input("Course (encoded)", min_value=0, step=1)

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        # Siapkan input data
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

        # Konversi ke DataFrame
        df = pd.DataFrame([input_data])

        # Prediksi
        prediction = model.predict(df)[0]
        result = '❌ Dropout' if prediction == 1 else '✅ Not Dropout'

        # Tampilkan hasil
        st.success(f"🎯 **Prediction Result:** {result}")
        st.write("Model memprediksi berdasarkan data akademik dan profil mahasiswa.")
