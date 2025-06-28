import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib

# Veriyi oku
df = pd.read_csv('movies_cleaned_features.csv')

# Bağımsız ve bağımlı değişkenler
X = df[['log_budget', 'budget_per_minute', 'vote_weighted', 'budget_popularity', 'popularity', 'runtime', 'vote_average']]
y = df['log_revenue']

# Train-Test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Hiperparametre aralıkları
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,  # 30 kombinasyon denesin
    cv=5,
    verbose=2,
    n_jobs=-1,
    scoring='r2',
    random_state=42
)

# Eğitimi başlat
random_search.fit(X_train, y_train)

# En iyi modeli al
best_model = random_search.best_estimator_

# Test seti üzerinde tahmin
y_pred = best_model.predict(X_test)

# Performans ölçümü
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Tuned XGBoost R² Skoru: {r2:.4f}")
print(f"Tuned XGBoost RMSE: {rmse:.4f}")
print(f"Best Parameters: {random_search.best_params_}")

# Modeli kaydet
joblib.dump(best_model, 'xgb_movie_revenue_model_tuned.pkl')
print("Tuned XGBoost modeli kaydedildi: xgb_movie_revenue_model_tuned.pkl")
