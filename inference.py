import streamlit as st 
from utils import *

st.title("Complete Soil & Climat dataset Processing")
st.subheader("Choose the dataset")
dataset_name = st.selectbox("dataset", ["Soil Dataset","Climat Dataset" , "Integrated dataset"])

st.subheader("Choose your operation")
processing_method = st.selectbox("operation", ["","Visualisation (Simple)","Visualisation (Complex)"  , "Update Operation" , "Delete Operation" , "Generale information" ,"Reduction","Discretization","Integration","Normalization" ,"Otliers Operations"])

if dataset_name:
    df , description = visualize(dataset_name)

if processing_method == "Visualisation (Simple)": 
    print(len(df))   
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
       visualize_complex(df , "Histogramme" )
    else: 
       visualize_complex(df , plot_type)

if processing_method == "Update Operation":
    st.subheader("Choose the attribute you wanna update")
    df = update_operation(df)
    st.dataframe(df)

if processing_method == "Delete Operation":
    st.subheader("Choose the attribute you wanna Delete")
    df = delete_operation(df)
    st.dataframe(df)

if processing_method == "Generale information":
    generale_informations(df)

if processing_method == "Normalization":
    st.subheader("Choose the Method you wanna use please!")
    method = st.selectbox("Method" , ["","MinMax" , "Z-Score"])
    if method == "MinMax":
        df = MinMax_Normalization(df)
        st.dataframe(df)
    elif method=="Z-Score" :
        df = ZScore_Normalization(df)
        st.dataframe(df)

if processing_method == "Reduction":
    st.subheader("Choose the Method you wanna use please!")
    method = st.selectbox("Method" , ["","Vertical" , "Horizental" , "Seasons"])
    if method == "Vertical":
        df = delete_operation(df)
        st.dataframe(df)

    elif method=="Horizental" :
        df = reduction_horiz(df)
        st.dataframe(df)

    elif method =="Seasons":
        if dataset_name != "Climat Dataset":
            st.error("This reduction method is not allowed for this dataset")
        else : 
            df = data_reduction(df )
            st.dataframe(df)


if processing_method == "Discretization":
    st.subheader("Choose the Method you wanna use please!")
    method = st.selectbox("Method" , ["","Amplitude" , "Equal-Frequancy"])
    if method == "Amplitude":
        df = Amplitude(df)
        st.dataframe(df)
    elif method == "Equal-Frequancy":
        df = discritize(df)
        st.dataframe(df)

if processing_method == "Integration":
    pass

if processing_method == "Otliers Operations":
    pass