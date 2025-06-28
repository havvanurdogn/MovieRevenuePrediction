import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Modeli yükle
with open('xgb_movie_revenue_model_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Veriyi yükle
df = pd.read_csv('movies_cleaned_features.csv')  # Senin temizlenmiş veri dosyan

# Özellikler ve hedef değişken
X = df.drop('revenue', axis=1)
y = df['revenue']

# Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Scatter Plot
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Gerçek Revenue')
plt.ylabel('Tahmin Edilen Revenue')
plt.title('Gerçek vs Tahmin Edilen Revenue (XGBoost)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # y=x çizgisi
plt.grid()
plt.show()
