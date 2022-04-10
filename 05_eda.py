import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web app ka title:
st.markdown('''
# **Exploratory Data Analysis Web Application**
This app is developed by Codanics youtube channel called **EDA App**
 ''')

# How to upload a file from PC

with st.sidebar.header("Upload your dataset (.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](df)")
#https://www.kaggle.com/c/feedback-prize-2021

# Profiling report for pandas
if uploaded_file is not None:
    @st.cache()
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative= True)
    st.header('**Input Df**')
    st.write(df)
    st.write('---')
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting for csv file, upload kar bhi do ab')
    if st.button('Press to use example data'):
        # example dataset

        @st.cache()
        def load_data():
            a = pd.DataFrame(np.random.randn(100,5),
                                columns=['age','banana','codanics','delta', 'ear'])
            return a
        df = load_data()
        pr = ProfileReport(df,explorative=True)
        st.header('**Input Dataframe**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

### Assignment: 1- write example of raw data set
# 2- Draw graphs of covid data
