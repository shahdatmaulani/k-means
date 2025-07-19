# src/visualization.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualization_page():
    st.title("📊 Data Visualization Sample")
    st.write("This is a sample visualization using Seaborn.")

    # Sample Data
    df = sns.load_dataset("tips")

    st.write("Dataset Sample:", df.head())

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="day", ax=ax)
    st.pyplot(fig)
