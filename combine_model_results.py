# combine_model_results.py

import pandas as pd

# 1. Tüm model sonuçlarını oku
linear = pd.read_csv('model_results_linear.csv')
decision_tree = pd.read_csv('model_results_decision_tree.csv')
xgboost = pd.read_csv('model_results_xgboost.csv')
lightgbm = pd.read_csv('model_results_lightgbm.csv')

# 2. Hepsini birleştir
all_results = pd.concat([linear, decision_tree, xgboost, lightgbm], ignore_index=True)

# 3. Büyükten küçüğe doğru (en iyi R² Score'a göre)
all_results = all_results.sort_values(by='R2 Score', ascending=False)

# 4. CSV'ye kaydet
all_results.to_csv('model_results.csv', index=False)

print('✅ Bütün model sonuçları birleştirildi ve model_results.csv oluşturuldu.')
