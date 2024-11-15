import streamlit as st 
from utils import *

st.title("Complete Soil & Climat dataset Processing")
st.subheader("Choose the dataset")
dataset_name = st.selectbox("dataset", ["Soil Dataset","Climat Dataset" , "Integrated dataset"])

st.subheader("Choose your operation")
processing_method = st.selectbox("operation", ["","Visualisation (Simple)","Visualisation (Complex)"  , "Update Operation" , "Delete Operation" , "Generale information" ,"Reduction","Discretization","Integration","Otliers Operations"])

if processing_method == "Visualisation (Simple)":
    df , description = visualize(dataset_name)
    st.markdown(f"### 1. DataFrame of {dataset_name} : ")
    st.dataframe(df)
    st.markdown("### 2. Description : ")
    st.write(f"{description}")
    if dataset_name == "Soil Dataset":
        st.markdown("### 3. Sneek Peak into the Areas")
        plot_polygons_on_map(df["geometry"])

if processing_method == "Visualisation (Complex)":
    st.subheader("Choose the plotting method ")
    plot_type = st.selectbox("plot_type", ["Scatter","HeatMap" ,"Histogramme"])
    if plot_type == "Histogramme":
       visualize_complex(dataset_name , "Histogramme" )
    else: 
       visualize_complex(dataset_name , plot_type)

if processing_method == "Update Operation":
    pass

if processing_method == "Delete Operation":
    pass

if processing_method == "Generale information":
    pass

if processing_method == "Reduction":
    pass

if processing_method == "Discretization":
    pass

if processing_method == "Integration":
    pass

if processing_method == "Otliers Operations":
    pass