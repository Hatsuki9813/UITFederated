import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide"
)

# CSS tùy chỉnh
st.markdown(
    """
    <style>
        div [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]{
            background-color: #1e293b;
            color: #f8f8f2 !important;
            border-radius: 8px;
            padding: 15px !important;
            padding-bottom: 25px !important;
            margin-bottom: 15px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }
        .session-block {
            background-color: #1e293b; /* Màu nền tối */
            color: #f8f8f2; /* Màu chữ sáng */
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .status-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .status-label {
            background-color: #64748b; /* Màu xám cho nhãn trạng thái */
            color: #f8f8f2;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
            margin-right: 10px;
        }
        .status-running {
            background-color: #38bdf8; /* Xanh dương cho Running */
        }
        .status-completed {
            background-color: #84cc16; /* Xanh lá cho Completed */
        }
        .status-failed {
            background-color: #f43f5e; /* Đỏ cho Failed */
        }
        .session-id {
            font-size: 0.8em;
            color: #cbd5e1;
        }
        .session-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .session-info {
            display: flex;
            margin-bottom: 5px;
            font-size: 0.95em;
        }
        .info-label {
            color: #cbd5e1;
            width: 70px;
            flex-shrink: 0;
        }
        .progress-bar-container {
            background-color: #475569;
            border-radius: 5px;
            height: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar {
            background-color: #38bdf8; /* Màu xanh dương cho tiến độ */
            height: 100%;
            border-radius: 5px;
        }
        .progress-text {
            font-size: 0.8em;
            color: #cbd5e1;
            text-align: right;
            margin-top: 5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dashboard")

# Container cho Quick Statistics (giữ nguyên)
with st.container():
    st.subheader("Quick Statistics")
    # ... (mã thống kê nhanh của bạn)

st.subheader("Training Sessions")

training_data = [
    {"ID": "FL-2023-04-15-001", "Status": "Running", "Title": "CNN Model Training", "Model": "CNN-LSTM", "Dataset": "Drebin-215", "Clients": 8, "Progress": 0.3},
    {"ID": "FL-2023-04-14-003", "Status": "Completed", "Title": "BERT Malware Detection", "Model": "BERT-Tiny", "Dataset": "AndroZoo-2020", "Clients": 12, "Progress": 1.0},
    {"ID": "FL-2023-04-14-002", "Status": "Failed", "Title": "GNN Experiment", "Model": "GNN-Basic", "Dataset": "Drebin-215", "Clients": 6, "Progress": 0.2},
    {"ID": "FL-2023-04-13-005", "Status": "Running", "Title": "Transformer Test", "Model": "Transformer-Small", "Dataset": "MalGenome", "Clients": 10, "Progress": 0.7},
    {"ID": "FL-2023-04-12-008", "Status": "Pending", "Title": "RNN Evaluation", "Model": "RNN-Attention", "Dataset": "AndroZoo-2020", "Clients": 5, "Progress": 0.0},
    {"ID": "FL-2023-04-11-012", "Status": "Completed", "Title": "MLP Baseline", "Model": "MLP-3Layer", "Dataset": "MalGenome", "Clients": 4, "Progress": 1.0},
]

num_cols = 4
cols = st.columns(num_cols)
col_index = 0

for session in training_data:
    with cols[col_index % num_cols]:
        with st.container():
            st.markdown(f'<div class="status-bar"><span class="status-label status-{session["Status"].lower()}">{session["Status"]}</span> <span class="session-id">ID: {session["ID"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-title">{session["Title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-info"><span class="info-label">Model:</span> <div>{session["Model"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-info"><span class="info-label">Dataset:</span> <div>{session["Dataset"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="session-info"><span class="info-label">Clients:</span> <div>{session["Clients"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="progress-bar-container"><div class="progress-bar" style="width: {session["Progress"] * 100}%;"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="progress-text">Progress: {int(session["Progress"] * 10)}/10 rounds</div>', unsafe_allow_html=True)
    col_index += 1

st.markdown("---")
st.markdown("Detailed information about training sessions will be updated continuously.")