# train_linear_regression.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

# 1. Veri Yükleme
X = pd.read_csv('movies_final_features.csv')
y = pd.read_csv('movies_final_target.csv')

# 2. Train-Test Bölme
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi
model = LinearRegression()
model.fit(X_train, y_train.values.ravel())

# 4. Tahmin
y_pred = model.predict(X_test)

# 5. Değerlendirme
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression RMSE: {rmse}')
print(f'Linear Regression R2 Score: {r2}')

# 6. Sonuçları CSV'ye Kaydet
results = pd.DataFrame({
    'Model': ['Linear Regression'],
    'RMSE': [rmse],
    'R2 Score': [r2]
})
results.to_csv('model_results_linear.csv', index=False)

# 7. Modeli Kaydet
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 8. Prediction Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Gerçek Revenue')
plt.ylabel('Tahmin Edilen Revenue')
plt.title('Gerçek vs Tahmin Edilen Revenue (Linear Regression)')
plt.savefig('prediction_plot_linear.png')
plt.close()
