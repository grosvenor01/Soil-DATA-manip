import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import streamlit as st
import plotly.graph_objects as go
import re
from shapely.wkt import loads

def plot_polygons_on_map(wkt_polygons):
    fig = go.Figure()
    all_coords = []
    for i, wkt_polygon in enumerate(wkt_polygons):
        try:
            polygon = loads(wkt_polygon)
            coords = np.array(polygon.exterior.coords)
            if coords.size > 0:
                lons, lats = coords[:, 0], coords[:, 1]
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    fill='toself',
                    name=f'Polygon {i+1}',
                    text=[f'Polygon {i+1}'],
                    marker=dict(size=10),
                    showlegend=True
                ))
                all_coords.extend(coords)
        except Exception as e:
            pass
    # Calculate center based on all coordinates
    if all_coords:
        center_lat = np.mean(np.array(all_coords)[:, 1])
        center_lon = np.mean(np.array(all_coords)[:, 0])
    else:
        center_lat = 37 
        center_lon = 9  

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8
        ),
        width=1000,
        height=800,
    )
    st.plotly_chart(fig)

def visualize(df_name):
    if df_name == "Soil Dataset":
        df = pd.read_excel("soil_dz_allprops.xlsx")
        fully_report = """This Algerian soil dataset comprises chemical analyses of topsoil and subsoil samples from various locations, spatially referenced by polygon geometries. The data includes percentages of 13 components for each layer: sand, silt, clay, pH (water), organic carbon (OC), nitrogen (N), base saturation (BS), cation exchange capacity (CEC), clay CEC, calcium carbonate (CaCO3), bulk density (BD), and the C/N ratio. each and the dataset provide the % of each one in top soil and sub soil in diffrent areas"""
    elif df_name == "Climat Dataset":
        df = pd.read_csv("finals.csv")
        fully_report = """This dataset contains hourly meteorological data for Algeria during 2019, encompassing surface pressure, rainfall, snowfall, air temperature, specific humidity, and wind speed/vector. Visual analysis reveals negligible snowfall throughout the year. The average accumulated values for air temperature, wind speed, and surface pressure across the four seasons """
    elif df_name =="Integrated Dataset":
        pass # we pass for now 
    else :
        return -1  # Error Case
    return df , fully_report

def plot_scatter(df , atr1 , atr2):
    fig = plt.figure(figsize=(6,6))
    plt.scatter(df[atr1], df[atr2], c=df[atr2], cmap='viridis', s=50, alpha=0.7)
    plt.xlabel(atr1)
    plt.ylabel(atr2)
    plt.title(f"Scatter Plot of {atr1} vs {atr2}")
    return fig

def plot_hitogramme(df , column):
    fig = plt.figure(figsize = (6,6))
    plt.xlabel(column)
    plt.title(f"Histogramme visualization pour l'atr : {column}")
    plt.hist(df[column], bins=10, edgecolor='black', linewidth=1.2)
    return fig
    
def visualize_complex(df_name, display_method , atr=None):
    df, _ = visualize(df_name) 
    figs = []
    if display_method == "Scatter":
        columns = df.columns
        for i in range(1, len(columns) - 1, 2):
            figs.append(plot_scatter(df, columns[i], columns[i + 1]))

    elif display_method == "Histogramme":
        attribute = st.selectbox("attribute", df.columns) 
        plot_hitogramme(df,columns[attribute])

    num_figs = len(figs)
    cols = st.columns(3)  
    fig_index = 0
    for i in range(num_figs):
        with cols[fig_index % 3]:
            st.pyplot(figs[i])
        fig_index += 1
    

    