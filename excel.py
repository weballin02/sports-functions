import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set the directory where Excel files are stored
EXCEL_DIRECTORY = "/Users/matthewfox/sports_functions/files"  # Update with your directory path

st.set_page_config(page_title="Dynamic Excel Viewer", page_icon="📊", layout="wide")

# Header with Apple-inspired design
st.markdown("<h1 style='text-align: center; color: #333333;'>📊 Dynamic Excel Viewer</h1>", unsafe_allow_html=True)

# Sidebar for file selection
st.sidebar.header("Choose an Excel File:")
excel_files = [f for f in os.listdir(EXCEL_DIRECTORY) if f.endswith('.xlsx') or f.endswith('.xls')]

if not excel_files:
    st.sidebar.write("No Excel files found.")
else:
    selected_file = st.sidebar.selectbox("Select an Excel file", excel_files)

    if selected_file:
        file_path = os.path.join(EXCEL_DIRECTORY, selected_file)
        
        # Load the Excel file and get available sheets
        excel_data = pd.ExcelFile(file_path)
        sheet = st.sidebar.selectbox("Select a Sheet", excel_data.sheet_names)

        # Read the selected sheet into a DataFrame
        try:
            data = pd.read_excel(excel_data, sheet_name=sheet)
            
            # Filter out rows where 'Game ID' is missing
            if 'Game ID' in data.columns:
                data = data.dropna(subset=['Game ID'])
            else:
                st.warning("No 'Game ID' column found in the data.")

            # File name and sheet header
            st.markdown(f"<h2 style='text-align: center; color: #666666;'>{selected_file} - {sheet}</h2>", unsafe_allow_html=True)
            
            # Display data table
            st.dataframe(data, use_container_width=True)

            # Display KPIs
            st.write("### Quick Stats:")
            st.write(f"Total Rows with Game ID: {data.shape[0]}")
            st.write(f"Total Columns: {data.shape[1]}")

            # Handle different column types
            numeric_columns = data.select_dtypes(include='number').columns.tolist()
            text_columns = data.select_dtypes(include='object').columns.tolist()

            # Chart options based on detected column types
            if st.checkbox("Show Summary Charts"):
                if numeric_columns:
                    selected_num_column = st.selectbox("Select a Numeric Column for Visualization", numeric_columns)
                    fig_num = px.histogram(data, x=selected_num_column, title=f"Distribution of {selected_num_column}", template="plotly_white")
                    st.plotly_chart(fig_num, use_container_width=True)
                if text_columns:
                    selected_text_column = st.selectbox("Select a Text Column for Visualization", text_columns)
                    fig_text = px.histogram(data, x=selected_text_column, title=f"Distribution of {selected_text_column}", template="plotly_white")
                    st.plotly_chart(fig_text, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.write("This file may not have a compatible structure for viewing.")

# Hide Streamlit menu and footer for clean look
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)