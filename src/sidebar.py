import streamlit as st
from streamlit_option_menu import option_menu
from src.config import SIDEBAR_OPTIONS, SIDEBAR_STYLES


def render_sidebar():
    """Render sidebar menu menggunakan konfigurasi dari config.py."""
    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=list(SIDEBAR_OPTIONS.keys()),
            icons=[SIDEBAR_OPTIONS[o] for o in SIDEBAR_OPTIONS],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles=SIDEBAR_STYLES
        )
    return selected
