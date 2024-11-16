import streamlit as st 
from utils import *

st.title("Complete Soil & Climat dataset Processing")

if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "description" not in st.session_state:
    st.session_state.description = None

st.subheader("Choose the dataset")
dataset_name = st.selectbox("dataset", ["Soil Dataset", "Climat Dataset", "Integrated Dataset"])

if dataset_name != st.session_state.dataset_name: #Load data only if dataset changed
    st.session_state.df, st.session_state.description = visualize(dataset_name)
    st.session_state.dataset_name = dataset_name


st.subheader("Choose your operation")
processing_method = st.selectbox("operation", ["", "Visualisation (Simple)", "Visualisation (Complex)", "Update Operation", "Delete Operation", "Generale information", "Reduction", "Discretization", "Integration", "Normalization", "Otliers Operations"])


if st.session_state.df is not None: #Check if a dataframe is loaded
    if processing_method:
        if processing_method == "Visualisation (Simple)":
            st.markdown(f"### 1. DataFrame of {dataset_name}:")
            st.dataframe(st.session_state.df)
            st.markdown("### 2. Description:")
            st.write(f"{st.session_state.description}")
            if dataset_name == "Soil Dataset":
                st.markdown("### 3. Sneek Peak into the Areas")
                plot_polygons_on_map(st.session_state.df["geometry"])

        elif processing_method == "Visualisation (Complex)":
            st.subheader("Choose the plotting method")
            plot_type = st.selectbox("plot_type", ["Scatter", "HeatMap", "Histogramme"])
            visualize_complex(st.session_state.df, plot_type)

        elif processing_method == "Update Operation":
            st.session_state.df = update_operation(st.session_state.df)  # Update in session state
            st.dataframe(st.session_state.df)

        elif processing_method == "Delete Operation":
            st.session_state.df = delete_operation(st.session_state.df)  # Update in session state
            st.dataframe(st.session_state.df)

        elif processing_method == "Generale information":
            generale_informations(st.session_state.df)
        
        elif processing_method == "Normalization":
            method = st.selectbox("Method", ["", "MinMax", "Z-Score"])
            if method == "MinMax":
                st.session_state.df = MinMax_Normalization(st.session_state.df)
                st.dataframe(st.session_state.df)
            elif method == "Z-Score":
                st.session_state.df = ZScore_Normalization(st.session_state.df)
                st.dataframe(st.session_state.df)
                
        elif processing_method == "Reduction":
            st.subheader("Choose the Method you wanna use please!")
            method = st.selectbox("Method", ["", "Vertical", "Horizental", "Seasons"])
            if method == "Vertical":
                st.session_state.df = delete_operation(st.session_state.df)  # Using delete_operation as a placeholder
                st.dataframe(st.session_state.df)
            elif method == "Horizental":
                st.session_state.df = reduction_horiz(st.session_state.df)
                st.dataframe(st.session_state.df)
            elif method == "Seasons":
                if dataset_name != "Climat Dataset":
                    st.error("This reduction method is not allowed for this dataset")
                else:
                    st.session_state.df = data_reduction(st.session_state.df)
                    st.dataframe(st.session_state.df)

        elif processing_method == "Discretization":
            st.subheader("Choose the Method you wanna use please!")
            method = st.selectbox("Method", ["", "Amplitude", "Equal-Frequancy"])
            if method == "Amplitude":
                st.session_state.df = Amplitude(st.session_state.df)
                st.dataframe(st.session_state.df)
            elif method == "Equal-Frequancy":
                st.session_state.df = discritize(st.session_state.df)
                st.dataframe(st.session_state.df)

        elif processing_method == "Integration":
            st.session_state.df = integration()
            st.dataframe(st.session_state.df)

        elif processing_method == "Otliers Operations":
            st.session_state.df = handling_outliers(st.session_state.df)
            st.dataframe(st.session_state.df)