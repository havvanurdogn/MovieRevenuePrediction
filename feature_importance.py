import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Modeli yükle
with open('lgbm_movie_revenue_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Özellik isimleri - Eğer X eğitimde kullandığın dataframe ise
# Buraya eğitimde kullandığın X değişkenindeki feature isimlerini gir
# Örneğin:
feature_names = feature_names = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_year', 'release_month']


# Feature importanceları al
importances = model.feature_importances_

# DataFrame oluştur
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Importancelara göre sırala
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Çizdir
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()
