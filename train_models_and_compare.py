# train_models_and_compare.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle

# 1. Veri Yükleme
X = pd.read_csv('movies_final_features.csv')
y = pd.read_csv('movies_final_target.csv')

# 2. Train-Test Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modeller
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

results = []

# 4. Model Eğitimi ve Değerlendirme
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'R2 Score': r2
    })

# 5. Sonuçları Tablo Olarak Kaydet
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)

# 6. En İyi Modeli Seç
best_model_name = results_df.sort_values(by='RMSE').iloc[0]['Model']
best_model = models[best_model_name]

# 7. En İyi Modeli Kaydet
with open('best_movie_revenue_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Model eğitimi tamamlandı. En iyi model: {best_model_name}")

# 8. Korelasyon Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Değişkenler Arası Korelasyon Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 9. Feature Importance (XGBoost için)
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train.values.ravel())

plt.figure(figsize=(10, 6))
importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)
plt.barh(X.columns[sorted_idx], importance[sorted_idx])
plt.title('Feature Importances')
plt.savefig('feature_importance.png')
plt.close()

# 10. Prediction Plot
best_model.fit(X_train, y_train.values.ravel())
y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Gerçek Revenue')
plt.ylabel('Tahmin Edilen Revenue')
plt.title('Gerçek vs Tahmin Edilen Revenue')
plt.savefig('prediction_plot.png')
plt.close()
