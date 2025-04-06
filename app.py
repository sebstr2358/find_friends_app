import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, 'r', encoding='UTF-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=";")
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


with st.sidebar:
    st.header('Powiedz nam coś o sobie')
    st.markdown('Pomożemy Ci znależć osoby podobne do Ciebie')
    age = st.selectbox('Przedział wiekowy', ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64'], index=None)
    edu_level = st.selectbox('Wykształcenie', ['Podstawowe', 'Średnie', 'Wyższe'], index=None)
    fav_animals = st.selectbox('Ulubione zwierzę', ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i psy'], index=None)
    fav_place = st.selectbox('Ulubione miejscw', ['Nad wodą', 'W lesie', 'W górach', 'Inne'], index=None)
    gender = st.radio('Płeć', ['Mężczyzna', 'Kobieta'], index=None)
        
    

person_df = pd.DataFrame([{
    "age": age,
    "edu_level": edu_level,
    "fav_animals": fav_animals,
    "fav_place": fav_place,
    "gender": gender,
    }])

if(age and edu_level and fav_animals and fav_place and gender):
    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()

    predicted_cluster_id = predict_model(model, data=person_df)['Cluster'].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]
    st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
    st.metric('Liczba Twoich znajomych', len(same_cluster_df))
    percent_clusters = round((len(same_cluster_df) / len(all_df) * 100))
    st.write(f"Stanowicie ok. {percent_clusters}% wszystkich ankietowanych!")

    st.header('Więcej o Twoich znajomych')
    min_value = min(10, len(same_cluster_df))
    st.subheader(f'{min_value} losowych wierszy')
    st.dataframe(
        same_cluster_df.sample(min_value),
        use_container_width=True,
        hide_index=True
    )
    fig = px.histogram(same_cluster_df.sort_values('age'), x='age')
    fig.update_layout(
        title='Rozkład wieku',
        xaxis_title='Wiek',
        yaxis_title='Liczba osób' 
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('edu_level'), x='edu_level')
    fig.update_layout(
        title='Wykształcenie osób z grupy',
        xaxis_title='Wykształcenie',
        yaxis_title='Liczba osób' 
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('fav_animals'), x='fav_animals')
    fig.update_layout(
        title='Ulubione zwierzę',
        xaxis_title='Zwierzę',
        yaxis_title='Liczba osób' 
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('fav_place'), x='fav_place')
    fig.update_layout(
        title='Ulubione miejsce wypoczynku',
        xaxis_title='Miejsce',
        yaxis_title='Liczba osób' 
    )
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df.sort_values('gender'), x='gender')
    fig.update_layout(
        title='Płeć',
        xaxis_title='Płeć',
        yaxis_title='Liczba osób' 
    )
    st.plotly_chart(fig)
else:
    st.markdown('<h2 style="color:red; fontsize:28px;">Powiedz coś o sobie...</h2>', unsafe_allow_html=True)
    st.markdown('<h5 style="color:red;">Wybierz jedną wartość w każdym z pól po lewej stronie, abyśmy mogli Cię zakwalifikować do jakiejś grupy.</h5>', unsafe_allow_html=True)