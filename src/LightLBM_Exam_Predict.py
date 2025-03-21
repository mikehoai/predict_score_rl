import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
data = pd.read_csv("diem_thi_thpt_2024.csv")
data.fillna(0, inplace=True)

# 2. Chuẩn bị dữ liệu
X = data[['Toan', 'NguVan', 'NgoaiNgu']]
y = data[['Toan']]

# 3. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình LightGBM với siêu tham số điều chỉnh
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    verbose=10
)

eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, eval_metric='rmse')

# Vẽ biểu đồ loss
results = model.evals_result_  # Corrected attribute name
epochs = len(results['valid_0']['rmse']) if 'valid_0' in results else 0
x_axis = range(0, epochs)
if epochs > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, results['valid_0']['rmse'], label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('LightGBM RMSE')
    plt.legend()
    plt.show()

# 5. Dự đoán
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_test['Toan'], y_pred))  # Chỉ so sánh với cột 'Toan'
print(f"LightGBM RMSE: {rmse}")

# 7. Dự đoán điểm dựa vào điểm trung bình năm ngoái
new_data = pd.DataFrame({'Toan': [6.45], 'NguVan': [7.23], 'NgoaiNgu': [5.51]})
predicted_scores = model.predict(new_data)
print(f"LightGBM dự đoán: Toán={predicted_scores[0]}")