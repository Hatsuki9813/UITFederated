import streamlit as st
import time
import random

uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

option = st.radio(
    "Chọn một mô hình:",
    ["MLP", "LSTM", "ResNet"]
)
import streamlit as st

# Giả sử đây là kết quả cuối cùng sau khi training
final_accuracy = 0.9123
final_loss = 0.1876
total_rounds = 10
training_time = "36.5s"

st.title("Local traning result")

st.metric(label="Accuracy", value=f"{final_accuracy:.4f}")
st.metric(label="Loss", value=f"{final_loss:.4f}")
st.write(f"Số vòng lặp: {total_rounds}")
st.write(f"Tổng thời gian training: {training_time}")

st.button("Start traning", type="primary")
