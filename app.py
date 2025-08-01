# app.py

import streamlit as st
from src.sidebar import render_sidebar
from src.visualization.page import visualization_page
from src.standardization.page import standardization_page

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="K-Means Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render sidebar dan ambil menu yang dipilih
menu_choice = render_sidebar()

# Tampilan konten utama sesuai menu
if menu_choice == "Data Visual":
    visualization_page()
elif menu_choice == "Z-Score":
    standardization_page()
else:
    st.title("Welcome to the K-Means Dashboard")
    st.markdown("Silakan pilih menu di sidebar untuk memulai.")

# Tampilkan footer
st.markdown("© 2025 K-Means Dashboard. All rights reserved.")