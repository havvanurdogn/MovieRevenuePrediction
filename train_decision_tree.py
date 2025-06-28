# train_decision_tree.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Veri Yükleme
X = pd.read_csv('movies_final_features.csv')
y = pd.read_csv('movies_final_target.csv')

# 2. Train-Test Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Eğitimi
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train.values.ravel())

# 4. Tahmin
y_pred = model.predict(X_test)

# 5. Değerlendirme
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'Decision Tree RMSE: {rmse}')
print(f'Decision Tree R2 Score: {r2}')

# 6. Sonuçları CSV'ye Kaydet
results = pd.DataFrame({
    'Model': ['Decision Tree'],
    'RMSE': [rmse],
    'R2 Score': [r2]
})
results.to_csv('model_results_decision_tree.csv', index=False)

# 7. Modeli Kaydet
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 8. Prediction Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Gerçek Revenue')
plt.ylabel('Tahmin Edilen Revenue')
plt.title('Gerçek vs Tahmin Edilen Revenue (Decision Tree)')
plt.savefig('prediction_plot_decision_tree.png')
plt.close()
