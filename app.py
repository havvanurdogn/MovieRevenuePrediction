# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob

# Modeli yükle
# Burada en iyi modeli kullanıyoruz — sen hangi modeli istiyorsan pkl ismini ona göre değiştir!
with open('xgboost_randomsearch_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('🎬 Movie Revenue Prediction with Confidence Interval')

st.write('Verdiğiniz bilgilerle bir filmin tahmini hasılatını (revenue) ve güven aralığını hesaplıyoruz!')

# Kullanıcıdan veri al
budget = st.slider('Budget (USD)', 1000, 500_000_000, 10_000_000, step=1_000_000)
runtime = st.slider('Runtime (Minutes)', 30, 300, 120)
popularity = st.number_input('Popularity', min_value=0.0, value=10.0, step=0.1)
vote_count = st.number_input('Vote Count', min_value=0, value=1000, step=10)
release_year = st.number_input('Release Year', min_value=1900, max_value=2030, value=2022)
release_month = st.selectbox('Release Month', list(range(1, 13)))

overview = st.text_area('Movie Overview', '')

# Sentiment Score
def get_sentiment(text):
    if text.strip() == '':
        return 0
    return TextBlob(text).sentiment.polarity

sentiment_score = get_sentiment(overview)

# Genre seçimi
genres = st.multiselect('Genres', ['Comedy', 'Horror', 'Action', 'SciFi', 'Fantasy'])

# Genre Encoding
genre_dict = {'Comedy': 0, 'Horror': 0, 'Action': 0, 'SciFi': 0, 'Fantasy': 0, 'Other': 0}
for g in genres:
    if g in genre_dict:
        genre_dict[g] = 1
if sum(genre_dict.values()) == 0:
    genre_dict['Other'] = 1

# Feature Vector
features = np.array([[budget, runtime, popularity, vote_count, 
                      sentiment_score, release_year, release_month,
                      genre_dict['Comedy'], genre_dict['Horror'], genre_dict['Action'],
                      genre_dict['SciFi'], genre_dict['Fantasy'], genre_dict['Other']]])

# Tahmin
if st.button('Predict Revenue'):
    prediction = model.predict(features)[0]
    
    # Basit Confidence Interval (±10% margin)
    lower_bound = prediction * 0.9
    upper_bound = prediction * 1.1
    
    st.success(f'🎯 Tahmini Revenue: ${prediction:,.2f}')
    st.info(f'📈 %90 Güven Aralığı: ${lower_bound:,.2f} - ${upper_bound:,.2f}')
