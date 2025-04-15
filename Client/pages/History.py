import streamlit as st
import pandas as pd
import numpy as np


# Dữ liệu mẫu
data = [
    {"ID": 1, "ClientName": "Client_A", "Metrics": "Loss: 0.21", "LocalAccuracy": 0.91},
    {"ID": 2, "ClientName": "Client_B", "Metrics": "Loss: 0.19", "LocalAccuracy": 0.88},
    {"ID": 3, "ClientName": "Client_C", "Metrics": "Loss: 0.17", "LocalAccuracy": 0.93},
]
serverdata = [
    { "ServerAccuracy": 0.95, "ServerLoss": 0.15, "Metrics":  0.21}
]

df = pd.DataFrame(data)
serverdf = pd.DataFrame(serverdata) 
# Hiển thị bảng
st.title("Client history")
st.table(df)
st.title("Server history")
st.table(serverdf)