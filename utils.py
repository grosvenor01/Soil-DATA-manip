import pandas as pd
import numpy as np 
import seaborn as sns 
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

def plot_scatter(df , atr1 , atr2):
    fig = plt.figure(figsize=(4,4))
    plt.scatter(df[atr1], df[atr2], c=df[atr2], cmap='viridis', s=50, alpha=0.7)
    plt.xlabel(atr1)
    plt.ylabel(atr2)
    plt.title(f"Scatter Plot of {atr1} vs {atr2}")
    return fig

def plot_histogramme(df , column):
    fig = plt.figure(figsize = (6,6))
    plt.hist(df[column], bins=10, edgecolor='black', linewidth=1.2)
    plt.xlabel(column)
    plt.title(f"Histogramme visualization pour l'atr : {column}")
    return fig
    
def plot_heatmap(df):
    df_new = df.copy()
    for i in df_new.columns:
        if df[i].dtype not in ["float64" , "int64"]:
            df_new = df_new.drop(columns=[i])
    fig = plt.figure(figsize=(4,4))
    sns.heatmap(df_new.corr())
    return fig

def get_season(date):
    month = int(date.split("-")[1])
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

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

def visualize_complex(df, display_method):
    figs = []
    columns = df.columns
    if display_method == "Scatter":
        attribute1 = st.selectbox("attribute1", columns)
        attribute2 = st.selectbox("attribute2", columns)
        percentage = st.selectbox("percentage", ["25%" , "50%" , "75%" , "100%"])

        if percentage == "100%":
            figs.append(plot_scatter(df, attribute1, attribute2))

        elif percentage:
            percentage = 25 if percentage == "25%" else (50 if percentage == "50%"  else 75)
            size_data = percentage * len(df) // 100
            new_df = df.sample(n=size_data , random_state=42)
            figs.append(plot_scatter(new_df, attribute1, attribute2))

    elif display_method == "Histogramme":
        attribute = st.selectbox("attribute", columns) 
        if attribute :
            st.pyplot(plot_histogramme(df,attribute))
    
    else :
        st.pyplot(plot_heatmap(df))
    
    num_figs = len(figs)
    cols = st.columns(3)  
    fig_index = 0
    for i in range(num_figs):
        with cols[fig_index % 3]:
            st.pyplot(figs[i])
        fig_index += 1
    
def update_operation(df):
    attribute = st.selectbox("attribute to modify", df.columns)
    index =  st.number_input("index" , 0 , len(df)-1 )
    new_value = st.text_input("new")
    try : 
        if df[attribute].dtype == "float64":
            new_value = float(new_value)
            df.loc[index , attribute] = new_value
            return df
        else :
            pass
    except Exception as e:
        st.error("Error occured or type mismatch"+str(e))

def delete_operation(df):
    attribute = st.selectbox("Type of delete", ["Column", "Index"])
    if attribute == "Column":
        columns_to_delete = st.multiselect("Select columns to delete", df.columns)
        if columns_to_delete:
            df = df.drop(columns=columns_to_delete)
    elif attribute == "Index":
        index_str = st.text_input("Enter index to delete")
        try:
            index = int(index_str)
            if 0 <= index < len(df):
                df = df.drop(index, axis=0)
            else:
                st.error("Index out of bounds")
        except ValueError:
            st.error("Invalid index. Please enter an integer.")
    return df

def generale_informations(df):
    columns = st.multiselect("Select columns to delete", df.columns)
    for i in columns:
        mean_value = df[i].mean()
        median_value = df[i].median()
        mode_value = df[i].mode()
        min_value = df[i].min()
        max_value = df[i].max()
        # Range of values
        range_value = max_value - min_value
        variance_value = df[i].var()
        std_value = df[i].std()
        quartiles = [df[i].quantile(0),df[i].quantile(0.25), df[i].median(),df[i].quantile(0.75), df[i].quantile(1)]
        quartile_deviation = df[i].quantile(0.75) - df[i].quantile(0.25)
        with st.expander(f"Information for {i}"):
            st.write(f"**Data Type:** {df[i].dtype}")
            st.write(f"**Mean Value:** {mean_value}")
            st.write(f"**Median Value:** {median_value}")
            st.write(f"**Mode Value:** {mode_value}")
            st.write(f"**Unique Values:** {df[i].unique()}")
            st.write(f"**Missing Values:** {df[i].isnull().sum()}")
            st.write(f"**Range Values:** {range_value}")
            st.write(f"**Variance:** {variance_value}")
            st.write(f"**Standred deviation:** {std_value}")
            st.write(f"**Quart1:** {quartiles[0]}\n **Quart2:** {quartiles[1]}\n **Quart3:** {quartiles[2]}\n **Quart4:** {quartiles[3]}\n **Quart5:** {quartiles[4]}")
            st.write(f"**Quartile Deviation:** {quartile_deviation}")

def MinMax_Normalization(df):
    df_minmax = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype == 'float64' and column not in ["lat","lon"]:
            bas = min(df[column])
            haut = max(df[column])
            gap = haut - bas
            df_minmax[column] = (df[column] - bas)/gap
    return df_minmax

def ZScore_Normalization(df):
    df_zscore = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype == 'float64' and column not in ["lat","lon"]:
            mean_value = df[column].mean()
            std_value = df[column].std()
            df_zscore[column] = ((df[column]) - mean_value)/std_value
    return df_zscore

def reduction_horiz(df):
    df_temp = df.drop_duplicates()
    dropped = len(df) - len(df_temp)
    print(f"Number of dropped values {dropped}")
    return df_temp

def data_reduction(df):
    df_copy = df.copy()
    df_copy['Season'] = df_copy['time'].apply(get_season)
    df_copy = df_copy.drop(columns=['time'])
    aggregation_dict ={}
    for i in df_copy.columns :
        if i !="Season" and i!="lat" and i!="lon" :
            aggregation_dict[i]= "mean"
    df_copy = df_copy.groupby(['Season', 'lat', 'lon']).agg(aggregation_dict).reset_index()
    return df_copy