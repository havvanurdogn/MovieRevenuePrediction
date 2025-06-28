import pandas as pd
import numpy as np
from textblob import TextBlob

# 1. Veriyi oku
df = pd.read_csv('movies_metadata.csv', low_memory=False)

# 2. Vote Count Temizliği
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)

# 3. Release Date'ten Year ve Month Çıkart
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year.fillna(df['release_date'].dt.year.median())
df['release_month'] = df['release_date'].dt.month.fillna(df['release_date'].dt.month.median())

# 4. Overview'den Sentiment Score
def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return 0
    return TextBlob(str(text)).sentiment.polarity

df['sentiment_score'] = df['overview'].apply(get_sentiment)

# 5. Genre'ları Feature'a Çevir
def map_genres(genres):
    if pd.isna(genres):
        return 'Other'
    if 'Comedy' in genres:
        return 'Comedy'
    elif 'Horror' in genres:
        return 'Horror'
    elif 'Action' in genres:
        return 'Action'
    elif 'Science Fiction' in genres or 'Sci-Fi' in genres:
        return 'SciFi'
    elif 'Fantasy' in genres:
        return 'Fantasy'
    else:
        return 'Other'

df['main_genre'] = df['genres'].apply(map_genres)

# One-hot encoding
df['Comedy'] = (df['main_genre'] == 'Comedy').astype(int)
df['Horror'] = (df['main_genre'] == 'Horror').astype(int)
df['Action'] = (df['main_genre'] == 'Action').astype(int)
df['SciFi'] = (df['main_genre'] == 'SciFi').astype(int)
df['Fantasy'] = (df['main_genre'] == 'Fantasy').astype(int)
df['Other'] = (df['main_genre'] == 'Other').astype(int)

# 6. Target Variable: Revenue
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

# 7. Final Feature Set
features = df[['vote_count', 'release_year', 'release_month', 'sentiment_score',
               'Comedy', 'Horror', 'Action', 'SciFi', 'Fantasy', 'Other']]

target = df['revenue']

# 8. CSV Dosyalarına Kaydet
features.to_csv('movies_final_features.csv', index=False)
target.to_csv('movies_final_target.csv', index=False)

print('✅ Feature engineering (ESKİ HALİ) tamamlandı ve dosyalar kaydedildi.')
