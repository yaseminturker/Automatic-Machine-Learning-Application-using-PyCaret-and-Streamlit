from operator import index
import streamlit as st
import plotly.express as px

import pycaret
#from pycaret.regression import *
#from pycaret.classification import *
from pycaret.clustering import *


import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import time

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image("https://datachannel.co/wp-content/uploads/2020/04/automated-machine-learning-platform-akira-ai.png")
    st.title("Automatic Machine Learning Application for *CLUSTERING*")
    choice = st.radio("Navigation", ["Welcome","Upload the Dataset","Profiling the Dataset","Modelling ML Models", "Download the Best Model"])
    #st.info("This application helps you explore your data and build ML models by using PyCaret and Streamlit.")


if choice == "Welcome":
    st.image("https://as1.ftcdn.net/v2/jpg/02/20/14/38/1000_F_220143804_fc4xRygvJ8bn8JPQumtHJieDN4ORNyjs.jpg")  
    st.info("This application helps you explore your data and build ML models by using **PyCaret** and **Streamlit**.")
    st.markdown("""So let's get started:\n\nFirst, you need to load the dataset by clicking **'Upload Your Dataset'** on the left navigation.\n\nThen click on **'Profiling the Dataset'** and see a detailed analysis of your dataset.\n\nThe next stage (by clicking on **'Modelling ML Models'**) is the ML models setup and compare by using *PyCaret*. The goal of the *PyCaret* package is to automate the major steps for evaluating and comparing machine learning algorithms for classification, regression and clustering. The main benefit of the library is that a lot can be achieved with very few lines of code and little manual configuration.\n\nIn the last stage, you will get the best model's python pickle(*.pkl) file.""")
    st.error('Note that this application will prepare just CLUSTERING models!!!', icon="🚨")

if choice == "Upload the Dataset":
    st.title("Upload Your Dataset")
    st.error('Note that this application will prepare CLUSTERING models when uploading the dataset.', icon="⚠️")
    file = st.file_uploader("Upload Your Dataset for **CLUSTERING**")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)


if choice == "Profiling the Dataset": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')


if choice == "Modelling ML Models":
    st.subheader("Dataframe")
    st.dataframe(df)
    if st.button('Run Modelling'):
        cul= pycaret.clustering.setup(df, normalize = True)
        cul_exp=pull()
        st.dataframe(cul_exp)

        kmeans= create_model('kmeans')
        e= evaluate_model(kmeans)
        e=pull()
        st.dataframe(e)

        plot_model(kmeans, plot = "distance", display_format="streamlit")
        plot_model(kmeans, plot= 'elbow', display_format="streamlit")
        plot_model(kmeans, plot= 'cluster', display_format="streamlit")
        plot_model(kmeans, plot= 'distribution', display_format="streamlit")
        plot_model(kmeans, plot= 'silhouette', display_format="streamlit")
        plot_model(kmeans, plot= 'tsne', display_format="streamlit")

        result= assign_model(kmeans)
        #st.dataframe(result)

        save_model(kmeans, 'best_CLUS_model')

        st.success("Best model's pickle file is ready!", icon='✅')
        st.balloons()


if choice == "Download the Best Model": 
    with open('best_CLUS_model.pkl', 'rb') as f:
        st.subheader("Click and download Best Model .pkl file") 
        st.download_button('Download Model', f, file_name="best_CLUS_model.pkl")