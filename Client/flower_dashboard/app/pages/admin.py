import streamlit as st
import pandas as pd
import pathlib
import requests
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Bạn cần đăng nhập để truy cập trang này.")
    st.stop()
st.set_page_config(
    layout="wide",
)

def load_css(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.html(f"<style>{f.read()}</style>")

css_path = pathlib.Path("assets/styles/admin.css")
load_css(css_path)

st.title("Admin Dashboard")
global_loss = 0.324
global_accuracy = 86.7

tab1, tab2, tab3 = st.tabs(["Overview", "Models", "Datasets"])

# Nội dung tab Overview
with tab1:
    st.subheader("Global Model Overview")
    option = st.selectbox("Select Model", ["FNN", "Logistic Regression"])

# Container to rerender based on selection
    with st.container():

        if option:
            try:
            # Gửi yêu cầu POST tới Flask API
                response = requests.post("http://127.0.0.1:5000/get-global-model", json={"model": option})

                if response.status_code == 200:
                    try:
                        data = response.json()
                        global_accuracy = float(data.get("GlobalAccuracy", 0))
                        global_loss = float(data.get("GlobalLoss", 0))

                        st.success("Lấy thông tin thành công!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Global Loss", value=f"{global_loss:.4f}")
                        with col2:
                            st.metric(label="Global Accuracy", value=f"{global_accuracy:.2f}%")
                    except Exception:
                        st.error("Lỗi: Dữ liệu trả về không hợp lệ hoặc không phải JSON.")
                else:
                    st.error(f"Lỗi từ server: {response.text}")
            except Exception as e:
                st.error(f"Lỗi kết nối tới API: {e}")



    # System Overview Stats
    with st.container():
        st.subheader("System Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="admin-stat-card">
                    <div class="stat-title">Total Users</div>
                    <div class="stat-value">128</div>
                    <div class="stat-change">+12% from last month</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="admin-stat-card">
                    <div class="stat-title">Active Users</div>
                    <div class="stat-value">64</div>
                    <div class="stat-change">+8% from last month</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="admin-stat-card">
                    <div class="stat-title">Total Sessions</div>
                    <div class="stat-value">256</div>
                    <div class="stat-change">+42% from last month</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="admin-stat-card">
                    <div class="stat-title">Active Sessions</div>
                    <div class="stat-value">24</div>
                    <div class="stat-change">+6% from last month</div>
                </div>
            """, unsafe_allow_html=True)

    # Recent Activity Table with Search
    with st.container():
        st.subheader("Recent Activity")
        st.caption("Recent user activity and system events.")
        search_query_activity = st.text_input("Search activity...", "")
        data_activity = {
            'User': ['john.doe@example.com'],
            'IP': ['192.168.129.123'],
            'Date': ['2023-04-15 09:24'],
            'Status': ['Success'],
            'Local loss': ['0.123'],
            'Local accuracy': ['0.95'],
            'Dataset': ['CICMaldroid'],
            'Model': ['MLP'],
        }
        df_activity = pd.DataFrame(data_activity)

        if search_query_activity:
            df_activity = df_activity[
                df_activity['User'].str.lower().str.contains(search_query_activity.lower()) |
                df_activity['Action'].str.lower().str.contains(search_query_activity.lower())
            ]

        st.dataframe(df_activity, hide_index=True)

# Nội dung tab Models
with tab2:
    st.subheader("Models Management")

    with st.container():
        col_add, col_export = st.columns([1, 1])
        with col_add:
            if st.button("➕ Add Model", key="add_model"):
                st.info("Functionality to add a new model will be implemented here.")
        with col_export:
            if st.button("⬇️ Export" , key="export_model"):
                st.info("Functionality to export models will be implemented here.")

    with st.container():
        search_query = st.text_input("Search models...", "")

    # Sample data for the models table
    data_models = {
        'Name': ['CNN-LSTM', 'BERT-Tiny', 'GNN-Basic', 'RNN-Attention', 'MLP-3Layer'],
        'Type': ['Hybrid', 'Transformer', 'Graph', 'Recurrent', 'Feedforward'],
        'Added': ['2023-03-15', '2023-02-28', '2023-01-10', '2022-12-05', '2022-11-20'],
        'Usage': ['42%', '28%', '15%', '8%', '7%'],
        'Actions': ['🗑️', '🗑️', '🗑️', '🗑️', '🗑️'] # Placeholder for delete action
    }
    df_models = pd.DataFrame(data_models)

    # Filter the DataFrame based on the search query
    if search_query:
        df_models = df_models[df_models['Name'].str.lower().str.contains(search_query.lower()) |
                                  df_models['Type'].str.lower().str.contains(search_query.lower())]

    st.dataframe(df_models, hide_index=True)

# Nội dung tab Datasets (chúng ta sẽ tạo giao diện cơ bản trước)
with tab3:
    st.subheader("Datasets Management")

    with st.container():
        col_add_dataset, col_export_dataset = st.columns([1, 1])
        with col_add_dataset:
            if st.button("➕ Add Dataset", key="add_dataset"):
                st.info("Functionality to add a new dataset will be implemented here.")
        with col_export_dataset:
            if st.button("⬇️ Export", key="export_dataset"):
                st.info("Functionality to export datasets will be implemented here.")

    with st.container():
        search_query_dataset = st.text_input("Search datasets...", "")

    data_datasets = {
        'Name': ['Drebin-215', 'AndroZoo-2020', 'MalGenome', 'VirusShare-2022'],
        'Samples': [15000, 24000, 8500, 12300],
        'Size': ['2.3 GB', '4.8 GB', '1.7 GB', '3.1 GB'],
        'Added': ['2023-02-10', '2023-01-15', '2022-12-20', '2022-11-05'],
        'Usage': ['38%', '32%', '18%', '12%'],
        'Actions': ['🗑️', '🗑️', '🗑️', '🗑️']
    }
    df_datasets = pd.DataFrame(data_datasets)

    if search_query_dataset:
        df_datasets = df_datasets[df_datasets['Name'].str.lower().str.contains(search_query_dataset.lower())]

    st.dataframe(df_datasets, hide_index=True)