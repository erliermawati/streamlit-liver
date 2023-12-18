import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat data dan melatih model
def load_data():
    # Gantilah path dengan path ke dataset Anda
    data_path = 'path/to/your/liver_dataset.csv'
    df = pd.read_csv(data_path)
    
    # Pisahkan fitur dan label
    X = df.drop('Liver_disease', axis=1)
    y = df['Liver_disease']
    
    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Latih model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Hitung akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Fungsi untuk tampilan aplikasi web
def main():
    st.title("Aplikasi Prediksi Penyakit Liver")

    # Memuat data dan melatih model
    model, accuracy = load_data()

    # Tampilkan informasi akurasi model
    st.write(f"Akurasi Model: {accuracy:.2%}")

    # Formulir untuk input pengguna
    st.sidebar.header("Input Pengguna")
    age = st.sidebar.slider("Usia", 0, 100, 25)
    total_bilirubin = st.sidebar.slider("Total Bilirubin", 0.0, 8.0, 1.0)
    direct_bilirubin = st.sidebar.slider("Direct Bilirubin", 0.0, 4.0, 0.5)
    alk_phosphate = st.sidebar.slider("Alkaline Phosphotase", 0, 300, 150)
    sgpt = st.sidebar.slider("SGPT", 0, 200, 75)
    sgot = st.sidebar.slider("SGOT", 0, 200, 35)
    total_proteins = st.sidebar.slider("Total Proteins", 2.0, 10.0, 6.0)
    albumin = st.sidebar.slider("Albumin", 1.0, 5.0, 3.0)
    albumin_globulin_ratio = st.sidebar.slider("Albumin/Globulin Ratio", 0.1, 2.5, 1.0)

    # Membuat prediksi berdasarkan input pengguna
    input_data = [[age, total_bilirubin, direct_bilirubin, alk_phosphate, sgpt, sgot, total_proteins, albumin, albumin_globulin_ratio]]
    prediction = model.predict(input_data)[0]

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.write("Pasien mungkin memiliki penyakit liver.")
    else:
        st.write("Pasien mungkin tidak memiliki penyakit liver.")

if __name__ == "__main__":
    main()
