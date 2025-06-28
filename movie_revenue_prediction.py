# movie_revenue_prediction.py

import pandas as pd

# CSV dosyasını oku
df = pd.read_csv('movies.csv')

# İlk 5 satırı gösterelim (kontrol amaçlı)
print(df.head())

# Önemli kolonları seçelim
selected_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'original_language']

df = df[selected_columns]

# Temizlik: 0 veya negatif değerleri olan satırları silelim
df = df[(df['budget'] > 0) & (df['revenue'] > 0) & (df['runtime'] > 0)]

# Eksik veri var mı kontrol edelim
print(df.isnull().sum())

# Temizlenmiş veri kümesinin istatistiklerine bakalım
print(df.describe())
