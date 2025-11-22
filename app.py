import streamlit as st
import pandas as pd
import numpy as np
import perprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.nonparametric.kde import KDEUnivariate


df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

df = perprocessor.preprocess(df , region_df)

st.sidebar.title('Olympics Analysis')
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')
user_menu = st.sidebar.radio(
    'Select an option',
    ('Medal Tally','Overall Analysis','Country_wise Analysis','Athlete_wise Analysis')
)

if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')
    years , country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox('Select Year',years)
    selected_country = st.sidebar.selectbox('Select Year', country)
    medal_tally = helper.fetch_medal_tally(df , selected_year , selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    edition = df['Year'].unique().shape[0]-1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title('Top statistics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Edition')
        st.title(edition)
    with col2:
        st.header('Cities')
        st.title(cities)
    with col3:
        st.header('Sports')
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Events')
        st.title(events)
    with col2:
        st.header('Athletes')
        st.title(athletes)
    with col3:
        st.header('Nations')
        st.title(nations)
    nation_over_time = helper.data_over_time(df ,'region')
    fig = px.line(nation_over_time , x ='Edition' , y = 'region')
    st.title('Participation over the years')
    st.plotly_chart(fig)

    event_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(event_over_time, x='Edition', y='Event')
    st.title('Event over the years')
    st.plotly_chart(fig)

    athletes = helper.data_over_time(df, 'Name')
    fig = px.line(athletes, x='Edition', y='Name')
    st.title('Athletes over the years')
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
        annot=True)
    st.pyplot(fig)

    st.title('Most successful Athletes')
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df , selected_sport)
    st.table(x)

if  user_menu == 'Country_wise Analysis':

    st.sidebar.title('Country wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select Country', country_list)

    country_df = helper.yearwise_medal_tally(df , selected_country)
    fig = px.line(country_df, x='Year', y='Medal')
    st.title(selected_country + ' Madal Tally over the years')
    st.plotly_chart(fig)

    pt = helper.country_event_heatmap(df , selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt, annot=True)
    st.title(selected_country + ' excel in the following sports')
    st.pyplot(fig)

    st.title('Top 10 Athletes of ' + selected_country)
    top10_df = helper.most_successful_athlete(df , selected_country)
    st.table(top10_df)

if user_menu == 'Athlete_wise Analysis':
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    datasets = [
        ("Overall Age", x1),
        ("Gold Medalist", x2),
        ("Silver Medalist", x3),
        ("Bronze Medalist", x4),
    ]

    fig = go.Figure()

    for label, data in datasets:

        if len(data) < 2:  # avoid errors if a medal category is empty
            continue

        kde = KDEUnivariate(data)
        kde.fit(bw="scott")  # smooth KDE curve like distplot

        x_range = np.linspace(data.min(), data.max(), 400)
        y_vals = kde.evaluate(x_range)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_vals,
            mode="lines",
            name=label
        ))

    fig.update_layout(
        title="Athlete wise Analysis",
        xaxis_title="Age",
        yaxis_title="Density",
        template="simple_white"
    )

    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']

    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        ages = temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna()

        if len(ages) > 1:  # KDE needs minimum 2 values
            x.append(ages)
            name.append(sport)

    fig = go.Figure()

    # Create KDE curve for each sport
    for ages, sport in zip(x, name):
        kde = KDEUnivariate(ages)
        kde.fit(bw="scott")

        x_range = np.linspace(ages.min(), ages.max(), 400)
        y_vals = kde.evaluate(x_range)

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_vals,
            mode="lines",
            name=sport
        ))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
        title="Distribution of Age wrt Sports (Gold Medalist)",
        xaxis_title="Age",
        yaxis_title="Density",
        template="simple_white"
    )

    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)

    temp_df = helper.weight_v_height(df, selected_sport)

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=temp_df,
        x='Weight',
        y='Height',
        hue='Medal',
        style='Sex',
        s=60,
        ax=ax
    )

    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)


