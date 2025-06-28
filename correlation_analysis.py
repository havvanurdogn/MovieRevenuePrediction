import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini oku
df = pd.read_csv('movies.csv')

# Önemli kolonları seçelim (aynı temizlik adımları)
selected_columns = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity']
df = df[selected_columns]
df = df[(df['budget'] > 0) & (df['revenue'] > 0) & (df['runtime'] > 0)]

# Korelasyon Matrisi
correlation_matrix = df.corr()

# Heatmap Çizimi
plt.figure(figsize=(10,8))  # Görsel boyutu
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

plt.title('Değişkenler Arası Korelasyon Heatmap')
plt.show()
