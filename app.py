import streamlit as st
from src.sidebar import render_sidebar
from src.visualization.page import visualization_page
from src.standardization.page import standardization_page


def main():
    st.set_page_config(
        page_title="K-Means Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    menu_choice = render_sidebar()

    if menu_choice == "Data Visual":
        visualization_page()
    elif menu_choice == "Z-Score":
        standardization_page()
    else:
        st.title("Welcome to the K-Means Dashboard")
        st.markdown("Silakan pilih menu di sidebar untuk memulai.")

    st.markdown("© 2025 K-Means Dashboard. All rights reserved.")


if __name__ == "__main__":
    main()
