import streamlit as st

pg = st.navigation([st.Page("ClientDetail.py"), st.Page("Training.py"), st.Page("History.py")])
pg.run()
