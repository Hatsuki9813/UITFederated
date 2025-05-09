import streamlit as st
import pathlib
import requests
import json


# Gọi một hàm từ module
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
def Login():
    if st.session_state.logged_in:
        Dashboard()
        st.stop()  # Dừng render phần còn lại

# Giao diện login/signup
    st.title("🔐 User Login")

    page = st.sidebar.selectbox("Navigation", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if page == "Login":
        if st.button("Login"):
            res = requests.post("http://localhost:5000/login", json={"username": username, "password": password})
            try:
                data = res.json()
                if data["status"] == "success":
                    st.session_state.logged_in = True
                    st.session_state.user_id = data["user_id"]
                    st.session_state.username = username
                    st.success("Đăng nhập thành công!")
                    st.rerun()
                else:
                    st.error(data["message"])
            except Exception as e:
                st.error("Lỗi khi xử lý phản hồi từ server.")

    elif page == "Sign Up":
        if st.button("Sign Up"):
            res = requests.post("http://localhost:5000/signup", json={"username": username, "password": password})
            try:
                data = res.json()
                if data["status"] == "success":
                    st.success("Đăng ký thành công!")
                else:
                    st.error(data["message"])
            except Exception as e:
                st.error("Lỗi khi xử lý phản hồi từ server.")
def Dashboard():
    with open("assets/logo.png", "rb") as f:
        icon_bytes = f.read()

    st.set_page_config(
        page_title="Flower Dashboard Demo",
        page_icon=icon_bytes,
        layout="wide"
    )

    css_path = pathlib.Path("assets/styles/dashboard.css")
    with open(css_path, encoding='utf-8') as f:
        st.html(f"<style>{f.read()}</style>")

    st.title("Dashboard")

    # Container cho Quick Statistics
    with st.container():
        st.subheader("Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
                <div class="quick-stat-card">
                    <div class="stat-title">Total Sessions</div>
                    <div class="stat-value">128</div>
                    <div class="stat-change">+12% from last month</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="quick-stat-card">
                    <div class="stat-title">Active Sessions</div>
                    <div class="stat-value">24</div>
                    <div class="stat-change">+8% from last month</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="quick-stat-card">
                    <div class="stat-title">Popular Model</div>
                    <div class="stat-value">CNN-LSTM</div>
                    <div class="stat-change">Used in 42% of sessions</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="quick-stat-card">
                    <div class="stat-title">Popular Dataset</div>
                    <div class="stat-value">Drebin-215</div>
                    <div class="stat-change">Used in 38% of sessions</div>
                </div>
            """, unsafe_allow_html=True)

    st.subheader("Training Sessions")

    training_data = [
        {"ID": "FL-2025-04-29-001", "Status": "Running", "Title": "Image Classification", "Model": "ResNet-50", "Dataset": "CIFAR-100", "Clients": 20, "Progress": 0.6},
        {"ID": "FL-2025-04-28-010", "Status": "Completed", "Title": "Object Detection", "Model": "YOLOv3", "Dataset": "COCO", "Clients": 15, "Progress": 1.0},
        {"ID": "FL-2025-04-28-005", "Status": "Failed", "Title": "Sentiment Analysis", "Model": "BERT-Base", "Dataset": "IMDb", "Clients": 25, "Progress": 0.1},
        {"ID": "FL-2025-04-27-012", "Status": "Running", "Title": "Language Modeling", "Model": "GPT-2", "Dataset": "WikiText-103", "Clients": 10, "Progress": 0.8},
        {"ID": "FL-2025-04-27-003", "Status": "Pending", "Title": "Speech Recognition", "Model": "DeepSpeech", "Dataset": "LibriSpeech", "Clients": 30, "Progress": 0.0},
        {"ID": "FL-2025-04-26-008", "Status": "Completed", "Title": "Machine Translation", "Model": "Transformer", "Dataset": "WMT16", "Clients": 18, "Progress": 1.0},
        {"ID": "FL-2025-04-26-001", "Status": "Cancelled", "Title": "Style Transfer", "Model": "Neural Style", "Dataset": "WikiArt", "Clients": 12, "Progress": 0.0},
        {"ID": "FL-2025-04-25-015", "Status": "Running", "Title": "Anomaly Detection", "Model": "Autoencoder", "Dataset": "KDD Cup 99", "Clients": 22, "Progress": 0.4},
        {"ID": "FL-2025-04-25-007", "Status": "Completed", "Title": "Reinforcement Learning", "Model": "DQN", "Dataset": "Atari", "Clients": 8, "Progress": 1.0},
        {"ID": "FL-2025-04-24-011", "Status": "Failed", "Title": "Generative Adversarial Network", "Model": "DCGAN", "Dataset": "MNIST", "Clients": 16, "Progress": 0.2},
        {"ID": "FL-2025-04-24-002", "Status": "Running", "Title": "Graph Neural Network", "Model": "GCN", "Dataset": "Cora", "Clients": 14, "Progress": 0.7},
        {"ID": "FL-2025-04-23-009", "Status": "Pending", "Title": "Time Series Forecasting", "Model": "LSTM", "Dataset": "Yahoo Finance", "Clients": 28, "Progress": 0.0},
        {"ID": "FL-2025-04-23-004", "Status": "Completed", "Title": "Clustering Analysis", "Model": "K-Means", "Dataset": "Iris", "Clients": 32, "Progress": 1.0},
        {"ID": "FL-2025-04-22-013", "Status": "Cancelled", "Title": "Dimensionality Reduction", "Model": "PCA", "Dataset": "Wine", "Clients": 19, "Progress": 0.0},
        {"ID": "FL-2025-04-22-006", "Status": "Running", "Title": "Federated Averaging", "Model": "FedAvg", "Dataset": "Fashion-MNIST", "Clients": 21, "Progress": 0.5},
        {"ID": "FL-2025-04-21-010", "Status": "Completed", "Title": "Ensemble Learning", "Model": "RandomForest", "Dataset": "Titanic", "Clients": 26, "Progress": 1.0},
    ]

    with st.container():
        col_search, col_filter, col_button = st.columns([3, 1, 1])

        with col_search:
            search_query = st.text_input(" ", placeholder="Enter model, dataset, or ID", key="search_query")

        with col_filter:
            status_filter = st.selectbox(
                " ",
                ["All", "Running", "Completed", "Failed", "Pending", "Cancelled"],
                key="status_filter",
            )

        with col_button:
            if st.button("New Training Session", key="new_session", use_container_width=True):
                st.switch_page("pages/Training.py")

        # --- Lọc dữ liệu dựa trên search và filter ---
        filtered_sessions = [
            session for session in training_data
            if (status_filter == "All" or session["Status"] == status_filter)
            and (
                search_query.lower() in session["ID"].lower()
                or search_query.lower() in session["Title"].lower()
                or search_query.lower() in session["Model"].lower()
                or search_query.lower() in session["Dataset"].lower()
            )
        ]

        num_cols = 5
        cols = st.columns(num_cols)
        col_index = 0

        for session in filtered_sessions:
            with cols[col_index % num_cols]:
                with st.container():
                    st.markdown(f"""
                        <div class="session-card">
                            <div class="status-bar">
                                <span class="status-label status-{session["Status"].lower()}">{session["Status"]}</span> 
                                <span class="session-id">ID: {session["ID"]}</span>
                            </div>
                            <div class="session-title">{session["Title"]}</div>
                            <div class="session-info">
                                <span class="info-label">Model:</span> <div>{session["Model"]}</div>
                            </div>
                            <div class="session-info">
                                <span class="info-label">Dataset:</span> <div>{session["Dataset"]}</div>
                            </div>
                            <div class="session-info">
                                <span class="info-label">Clients:</span> <div>{session["Clients"]}</div>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar" style="width: {session["Progress"] * 100}%;"></div>
                            </div>
                            <div class="progress-text">Progress: {int(session["Progress"] * 10)}/10 rounds</div>
                        </div>
                    """, unsafe_allow_html=True)
            col_index += 1

    st.markdown("---")
    st.markdown("Detailed information about training sessions will be updated continuously.")
    if st.button("Logout", key="logoutbutton"):
        st.session_state.logged_in = False
        st.rerun()
def main():
    if st.session_state.logged_in:
        Dashboard()
    else:
        Login()
if __name__ == "__main__":
    main()