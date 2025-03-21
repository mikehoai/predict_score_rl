import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt  # Add this import

# 1. Đọc dữ liệu và làm sạch dữ liệu
data = pd.read_csv("diem_thi_thpt_2024.csv")
data.fillna(0, inplace=True)

# 2. Chuẩn bị dữ liệu
X = data[['Toan', 'NguVan', 'NgoaiNgu']]
y = data[['Toan', 'NguVan', 'NgoaiNgu']]

# 3. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình XGBoost với siêu tham số điều chỉnh
eval_set = [(X_train, y_train), (X_test, y_test)]  # Add this line
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)  # Modify this line

# 5. Dự đoán
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost RMSE: {rmse}")

# 7. Dự đoán điểm dựa vào điểm trung bình năm ngoái
new_data = pd.DataFrame({'Toan': [6.45], 'NguVan': [7.23], 'NgoaiNgu': [5.51]})
predicted_scores = model.predict(new_data)
print(f"XGBoost dự đoán: Toán={predicted_scores[0][0]}, Ngữ Văn={predicted_scores[0][1]}, Ngoại Ngữ={predicted_scores[0][2]}")

# 8. Vẽ biểu đồ loss
results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

# Plot training and validation loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()