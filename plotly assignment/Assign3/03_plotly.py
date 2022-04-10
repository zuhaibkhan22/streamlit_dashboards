# Import libraries
from sys import maxsize
import streamlit as st
import plotly.express as px
import pandas as pd

# Import datasets
st.title("Plotly and Streamlit ko mila ky app banani hy")

df = px.data.gapminder()
st.write(df)
#st.write(df.head())

st.write(df.columns)

# Summary Statistics
st.write(df.describe())

# Data Management

year_option = df['year'].unique().tolist()
year = st.selectbox("Which year should we plot? ", year_option, 0)
#df = df[df['year']==year]

# Plotting
fig = px.scatter(df, x = 'gdpPercap', y = 'lifeExp', size='pop', color='country', hover_name='country',
                log_x = True, size_max= 55,range_x= [100,100000], range_y = [20,90],
                animation_frame='year',animation_group='country')
fig.update_layout(width=700, height = 400)
st.write(fig)

#Also we can make interactive graphs for continents, simple change in: color='continent' and hover etc


# st.markdown('**Assignment**: Implement plotly for other parameters and draw interactive graphs')

# df = px.data.gapminder()
# st.write(df.head())
# #st.write(df['gdpPercap'].max())

# # Data Management

# year_option = df['year'].unique().tolist()
# year = st.selectbox("Which year should we plot? ", year_option, 0)
# df = df[df['year']==year]

# # Plotting
# fig = px.scatter(df, x = 'year', y = 'gdpPercap', size='year', color='country', hover_name='country',
#                 log_x = True, size_max= 55,range_x= [1950,2008], range_y = [10,10000])
# fig.update_layout(width=800, height = 700)
# st.write(fig)


# ### Make assignment on some other data and use streamlit.