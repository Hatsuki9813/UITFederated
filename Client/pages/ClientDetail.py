import streamlit as st


with st.form("InfoForm"):
    st.write("Please fill out the following information:")
    client_name = st.text_input("Client Name")
    client_email = st.text_input("Client Email")
    client_phone = st.text_input("Client Phone Number")
    submit_button = st.form_submit_button(label="Submit")
