import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Veriyi oku ve temizle
df = pd.read_csv('movies.csv')

# Gelişmiş Feature listesi
selected_features = ['budget', 'vote_count', 'popularity', 'runtime', 'vote_average', 'revenue']

df = df[selected_features]
df = df[(df['budget'] > 0) & (df['revenue'] > 0) & (df['runtime'] > 0) & (df['vote_count'] > 0)]

# Bağımsız değişkenler (X) ve bağımlı değişken (y)
X = df[['budget', 'vote_count', 'popularity', 'runtime', 'vote_average']]
y = df['revenue']

# Train/Test bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model = LinearRegression()

# Model eğitimi
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans ölçümü
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"R² Skoru: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Modeli kaydet
joblib.dump(model, 'movie_revenue_model.pkl')
print("Model kaydedildi: movie_revenue_model.pkl")
