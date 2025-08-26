import streamlit as st
import pandas as pd
import joblib

loaded_model = joblib.load("mlbb_model.joblib")

st.title("Machine Learning Classification")
st.markdown("Ini adalah aplikasi untuk memprediksi jenis role penyerang atau bertahan di MLBB")

kill = st.slider("Jumlah Kill", 0, 20)
assist = st.slider("Jumlah Assist", 0, 20)
death = st.slider("Jumlah Death", 0, 20)
turret = st.slider("Jumlah Turret", 0, 20)

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[kill,assist,death,turret]], columns=["kill","assist","death","turret"])
	st.success(f"Hasil Prediksi : {loaded_model.predict(data_baru)[0]}")
	st.balloons()

st.caption("Dibuat dengan :heart: oleh Adi Setiawan")