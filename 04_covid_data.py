# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd

st.markdown('**Case Study**: Implementing plotly for Covid data worldwide and drawing interactive graphs')

df = pd.read_csv('owid-covid-data.csv')
#st.write(df.head(10))
#st.write(df['total_cases'].sample(1000))
# df['total_cases'].dropna()
# df['new_cases'].dropna()

st.write(df.head(10))


#Data Management


#total_cases_opt = df['total_cases'].unique().tolist()

continent_opt = df['continent'].unique().tolist()
continent = st.selectbox("Which continent should we plot? ", continent_opt, 0)
df = df[df['continent']==continent]
#st.write(df.head(10))



#cardiovasc_death_rate
#hover_name=''


# (1)
fig = px.scatter(df, x = 'location', y = 'total_deaths',color='location', hover_name='location',
                log_x = False, size_max=100, range_x=['Albania','Vatican'], range_y = [100,30000],)
fig.update_layout(width=900, height = 550)
st.write(fig)


# (2)
fig = px.scatter(df, x = 'total_cases', y = 'total_deaths',color='location', hover_name='location',
                log_x = False, size_max=1000, range_x=[10000,1000000], range_y = [500,45000],
                animation_frame='iso_code',animation_group='total_deaths')
fig.update_layout(width=900, height = 550)
st.write(fig)

# (3)

fig = px.scatter(df, x = 'iso_code', y = 'new_cases_per_million',color='location', hover_name='location',
                log_x = False, size_max=50, range_x=['ALB','VAT'], range_y = [100,15000],
                animation_frame='location',animation_group='new_cases_per_million')
fig.update_layout(width=900, height = 500)
st.write(fig)
# animation_frame='iso_code',animation_group='total_deaths'


# (4)

fig = px.scatter(df, x = 'location', y = 'total_cases',color='population', hover_name='population',)
#fig.update_layout(width=800, height = 700)
st.write(fig)

# I drew continent with total_cases and iso_code with total_cases and population with total cases


fig = px.scatter(df, x = 'aged_65_older', y = 'cardiovasc_death_rate',color='location', hover_name='location',
                log_x = False, size_max=10)
#fig.update_layout(width=800, height = 700)
st.write(fig)

# Plotting
# fig = px.scatter(df, x = df['continent'], y = df['location'], size= 'continent', color='continent', hover_name='continent',
#                 log_x = False, size_max= 300,range_x= [100,1000], range_y = [10,1000])
# fig.update_layout(width=800, height = 700)
# st.write(fig)



#fig.update_layout(width=800, height = 700)

#animation_frame='continent',animation_group='continent'