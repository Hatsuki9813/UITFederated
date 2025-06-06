import streamlit as st
import pathlib
import subprocess
import requests
import json
import os
import time
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_IP = "http://10.0.145.238:5000"
SUPERNODE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../../flowerclient/client.py"))
SUPERNODE_DIR = os.path.dirname(SUPERNODE_PATH)  # Lấy thư mục chứa client.py
CERTIFICATE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../../flowerclient/certificates/ca.crt"))
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Bạn cần đăng nhập để truy cập trang này.")
    st.stop()

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
        "name": "CIC Maldroid 2020",
        "desc": "Standard dataset with 139 features extracted from Android malware samples",
        "samples": "15,000",
        "size": "100MB"
    },
]

models = [
    {
        "name": "FNN",
        "type": "Hybrid",
        "desc": "Neural network model with 3 hidden layers",
        "acc": "92.5%"
    },

]

st.title("Training")
# === Information Section ===
st.subheader("Training Information")
st.text_input("Enter username", placeholder="Username", key="username")

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
if st.button("Load CA.crt from server"):
    getca = requests.get(f"{SERVER_IP}/getcertificate")
    if getca.status_code == 200:
        with open(CERTIFICATE_PATH, 'wb') as f:
            f.write(getca.content)
        print("Tải file thành công")
    else:
        print("Tải file thất bại")

if st.button("Update Client Info"):
            selected_model = st.session_state.selected_model
            selected_dataset = st.session_state.selected_dataset
            name = st.session_state.username
            print(f"selected_model: {selected_model}")
            print(f"selected_dataset: {selected_dataset}")
            print(f"name: {name} ")
            ACCURACY_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../../flowerclient/local_accuracy.json"))
            accuracy_last_modified_time = os.path.getmtime(ACCURACY_PATH)
            #start_time = time.time()
            #timeout = 180
            #while True:
                   # if time.time() - start_time > timeout:
                        #raise TimeoutError("File was not updated within timeout.")
                    #if os.path.getmtime(ACCURACY_PATH) != accuracy_last_modified_time:
            try:
                with open(ACCURACY_PATH, "r") as f:
                                accuracy_data = json.load(f)
                                local_accuraccy = accuracy_data["local_accuracy"] 
                                print(f"Local Accuracy: {local_accuraccy}")

                                #break
            except json.JSONDecodeError:
                    print("error reading local loss")

                    #else:
                        #time.sleep(0.1)
            LOSS_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../../../flowerclient/local_loss.json"))
            print(LOSS_PATH)
            #loss_last_modified_time = os.path.getmtime(LOSS_PATH)
            #while True: 
            #        if time.time() - start_time > timeout:
            #            raise TimeoutError("File was not updated within timeout.")
            #        if os.path.getmtime(LOSS_PATH) != loss_last_modified_time:
            try:
                with open(LOSS_PATH, "r") as f:
                    loss_data = json.load(f)
                    local_loss = loss_data["local_loss"]
                    print(f"Local Loss: {local_loss}")
        
                    #break
            except json.JSONDecodeError:
                    print("error reading local loss")

            current_model = requests.post(f"{SERVER_IP}/set-current-model", json={"model": selected_model})        
            update_client_info = requests.post(f"{SERVER_IP}/update-client-info", json={"model": selected_model, "dataset": selected_dataset, "local_loss": local_loss, "local_accuracy": local_accuraccy, "clientname": st.session_state.username})
if st.button("Start Training", key="start-training-button", disabled=not selected_enough):
        st.session_state.training_log = f"Training started using **{st.session_state.selected_model}** model on **{st.session_state.selected_dataset}** dataset."
        
        log_placeholder = st.empty()
        with st.spinner("Training in progress..."):
            try:
                print(SUPERNODE_PATH)
                process = subprocess.Popen(
                    ["python", "client.py"],
                cwd=SUPERNODE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

                log_lines = ""  # To accumulate and display full output
                for line in iter(process.stdout.readline, ''):
                    if line:
                        log_lines += line
                        log_placeholder.text(log_lines)  # Cập nhật nội dung log theo thời gian thực

                process.stdout.close()
                process.wait()

                if process.returncode == 0:
                    st.success("Training script completed successfully.")
                else:
                    st.error(f"Training script failed with return code {process.returncode}")

            except Exception as e:
                st.error(f"Error running script: {e}")
# Hiển thị log nếu có
if "training_log" in st.session_state:
    st.markdown(f"<div class='training-log'>{st.session_state.training_log}</div>", unsafe_allow_html=True)
