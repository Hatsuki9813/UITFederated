import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide"
)

st.write('''<style>
    [data-testid="stHorizontalBlock"]:has(div.PortMarker) [data-testid="stMarkdownContainer"] p { 
        margin: 0px 0px 0.2rem; 
        color: #ff0000;
    }        
    </style>''', unsafe_allow_html=True)


with st.container():
    INcol1, INcol2 = st.columns(2) 
    with INcol1:
            st.write('Test 1')
            st.write("""<div class='PortMarker'/>""", unsafe_allow_html=True)
    with INcol2:
            st.write('Test 2')