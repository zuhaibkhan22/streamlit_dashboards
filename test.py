import streamlit as st
import seaborn as sns

st.header("This video is brought to you by Codanics")
st.text("Kia Apko maza arha hy")

phool = sns.load_dataset('iris')

st.write(phool.head())
st.write(phool[['sepal_length','petal_length']])

st.bar_chart(phool['sepal_length'])
st.line_chart(phool['sepal_length'])