# src/sidebar.py
import streamlit as st
from streamlit_option_menu import option_menu

def render_sidebar():
    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=["Z-Score", "Data Visual"],
            icons=["cloud-upload", "bar-chart"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "5px", "background-color": "#003366"},
                "icon": {"color": "white", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#006699",
                },
                "nav-link-selected": {
                    "background-color": "#005580",
                },
            }
        )
    return selected
