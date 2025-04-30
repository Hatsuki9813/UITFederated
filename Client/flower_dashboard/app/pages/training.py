import streamlit as st
import pathlib

st.set_page_config(
    layout="wide",
)

def load_css(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.html(f"<style>{f.read()}</style>")

css_path = pathlib.Path("assets/styles/training.css")
load_css(css_path)

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# === Dataset Options ===
datasets = [
    {
        "name": "Drebin-215",
        "desc": "Standard dataset with 215 features extracted from Android apps",
        "samples": "15,000",
        "size": "2.3 GB"
    },
    {
        "name": "AndroZoo-2020",
        "desc": "Large-scale collection of Android apps from various markets",
        "samples": "24,000",
        "size": "4.8 GB"
    },
    {
        "name": "MalGenome",
        "desc": "Focused collection of malicious Android applications",
        "samples": "8,500",
        "size": "1.7 GB"
    },
]

models = [
    {
        "name": "CNN-LSTM",
        "type": "Hybrid",
        "desc": "Combines CNN for feature extraction with LSTM for sequence analysis",
        "acc": "92.5%"
    },
    {
        "name": "BERT-Tiny",
        "type": "Transformer",
        "desc": "Lightweight transformer model adapted for malware detection and classification",
        "acc": "94.1%"
    },
    {
        "name": "GNN-Basic",
        "type": "Graph",
        "desc": "Graph neural network for analyzing app component relationships and interactions",
        "acc": "91.8%"
    },
]

st.title("Training")

# === Dataset Options ===
st.subheader("Select Dataset")
col_datasets = st.columns(len(datasets))
for i, dataset in enumerate(datasets):
    with col_datasets[i]:
        selected = st.session_state.selected_dataset == dataset["name"]
        card_class = "selected" if selected else ""
        st.markdown(f"""
            <div class="training-card {card_class}">
                <div class="card-header">
                    <span class="card-type">Dataset</span>
                    <span class="card-name">{dataset["name"]}</span>
                </div>
                <div class="card-body">
                    <p>{dataset["desc"]}</p>
                    <p>Samples: <strong>{dataset["samples"]}</strong></p>
                    <p>Size: <strong>{dataset["size"]}</strong></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(
            "Selected" if selected else "Select Dataset",
            key=f"select_dataset_{i}",
            disabled=selected,
            help=None if selected else f"Chọn {dataset['name']}"
        ):
            st.session_state.selected_dataset = dataset["name"]

# === Model Options ===
st.subheader("Select Model")
col_models = st.columns(len(models))
for i, model in enumerate(models):
    with col_models[i]:
        selected = st.session_state.selected_model == model["name"]
        card_class = "selected" if selected else ""
        st.markdown(f"""
            <div class="training-card {card_class}">
                <div class="card-header">
                    <span class="card-type">{model["type"]}</span>
                    <span class="card-name">{model["name"]}</span>
                </div>
                <div class="card-body">
                    <p>{model["desc"]}</p>
                    <p>Accuracy: <strong>{model["acc"]}</strong></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(
            "Selected" if selected else "Select Model",
            key=f"select_model_{i}",
            disabled=selected,
            help=None if selected else f"Chọn {model['name']}"
        ):
            st.session_state.selected_model = model["name"]

# === Start Training Button ===
selected_enough = st.session_state.selected_dataset and st.session_state.selected_model

if st.button("Start Training", key="start-training-button", disabled=not selected_enough):
        st.session_state.training_log = f"🚀 Training started using **{st.session_state.selected_model}** model on **{st.session_state.selected_dataset}** dataset."

# Hiển thị log nếu có
if "training_log" in st.session_state:
    st.markdown(f"<div class='training-log'>{st.session_state.training_log}</div>", unsafe_allow_html=True)
