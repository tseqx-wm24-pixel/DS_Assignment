import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

st.title("📊 模型性能比较")

# 加载模型
lr = joblib.load("lr_model.pkl")
dt = joblib.load("dt_model.pkl")
rf = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb_model
}

# 假设你在 Notebook 里有测试集 X_test, y_test
# 👉 这里为了演示，先用一个模拟数据
X_test = np.array([[50, 2], [80, 3], [120, 3], [200, 4], [300, 5]])
y_test = np.array([100, 160, 240, 400, 600])

# 计算性能指标
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    results.append([name, r2, rmse, mae, mape])

df_results = pd.DataFrame(results, columns=["Model", "R²", "RMSE", "MAE", "MAPE"])

st.write("### 📋 性能对比表")
st.dataframe(df_results)

st.write("### 📈 可视化对比")
st.bar_chart(df_results.set_index("Model")[["R²", "RMSE", "MAE", "MAPE"]])
