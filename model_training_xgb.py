import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib

# Veriyi oku
df = pd.read_csv('movies_cleaned_features.csv')

# Bağımsız ve bağımlı değişkenler
X = df[['log_budget', 'budget_per_minute', 'vote_weighted', 'budget_popularity', 'popularity', 'runtime', 'vote_average']]
y = df['log_revenue']

# Veriyi Train/Test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# Modeli eğit
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test)

# Performans ölçümleri
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"XGBoost R² Skoru: {r2:.4f}")
print(f"XGBoost RMSE: {rmse:.4f}")

# Cross Validation skoru
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"5-Fold Cross-Validation R² Skoru (Average): {cv_scores.mean():.4f}")

# Modeli kaydet
joblib.dump(model, 'xgb_movie_revenue_model.pkl')
print("XGBoost modeli kaydedildi: xgb_movie_revenue_model.pkl")
