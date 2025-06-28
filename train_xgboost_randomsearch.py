# train_xgboost_randomsearch.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# 1. Veri Yükleme
X = pd.read_csv('movies_final_features.csv')
y = pd.read_csv('movies_final_target.csv')

# 2. Train-Test Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model ve Parametreler
model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# 4. Model Eğitimi
random_search.fit(X_train, y_train.values.ravel())

# 5. En iyi modeli al
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# 6. Tahmin
y_pred = best_model.predict(X_test)

# 7. Değerlendirme
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'XGBoost (RandomSearch) RMSE: {rmse}')
print(f'XGBoost (RandomSearch) R2 Score: {r2}')

# 8. Sonuçları CSV'ye Kaydet
results = pd.DataFrame({
    'Model': ['XGBoost (RandomSearch)'],
    'RMSE': [rmse],
    'R2 Score': [r2]
})
results.to_csv('model_results_xgboost.csv', index=False)

# 9. Modeli Kaydet
with open('xgboost_randomsearch_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# 10. Feature Importance Plot
importance = best_model.feature_importances_
sorted_idx = np.argsort(importance)

plt.figure(figsize=(10, 8))
plt.barh(X.columns[sorted_idx], importance[sorted_idx])
plt.title('Feature Importances (XGBoost RandomSearch)')
plt.xlabel('Importance')
plt.savefig('feature_importance_xgboost.png')
plt.close()

# 11. Prediction Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Gerçek Revenue')
plt.ylabel('Tahmin Edilen Revenue')
plt.title('Gerçek vs Tahmin Edilen Revenue (XGBoost RandomSearch)')
plt.savefig('prediction_plot_xgboost.png')
plt.close()
