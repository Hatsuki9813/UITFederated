import streamlit as st
import pathlib
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Bạn cần đăng nhập để truy cập trang này.")
    st.stop()

with open("assets/logo.png", "rb") as f:
    icon_bytes = f.read()

st.set_page_config(
    page_title="Flower Dashboard Demo",
    page_icon=icon_bytes,
    layout="wide"
)

def load_css(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.html(f"<style>{f.read()}</style>")

css_path = pathlib.Path("assets/styles/setting.css")
load_css(css_path)

st.title("Settings")

tab_account, tab_appearance = st.tabs(["Account", "Appearance"])

# Nội dung tab Account
with tab_account:
    st.subheader("Account Information")
    st.caption("Update your account information and preferences.")

    name = st.text_input("Name", "John Doe")
    email = st.text_input("Email", "john.doe@example.com", disabled=True)
    st.caption("Contact admin to change your email address.")

    if st.button("Save Changes", key="save_changes"):
        st.success("Account information updated!") # Placeholder

    st.subheader("Password")
    st.caption("Change your password.")

    current_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Change Password", key ="change_password"):
        if new_password == confirm_password:
            st.success("Password changed successfully!") # Placeholder
        else:
            st.error("New password and confirm password do not match.")

# Nội dung tab Appearance
with tab_appearance:
    st.subheader("Appearance")
    st.write("Options to customize the appearance of the application will be added here.")
    # Bạn có thể thêm các tùy chọn như chọn theme (light/dark), cỡ chữ, v.v.
    theme_options = ["Light", "Dark", "System"]
    selected_theme = st.selectbox("Theme", theme_options)
    st.write(f"Selected theme: {selected_theme}")

    font_size_options = ["Small", "Medium", "Large"]
    selected_font_size = st.selectbox("Font Size", font_size_options)
    st.write(f"Selected font size: {selected_font_size}")
