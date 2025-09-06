import streamlit as st
from joblib import load
import numpy as np

st.set_page_config(page_title="房价预测", page_icon="🏠")

st.title("🏠 房价预测系统")

# 选择模型
model_choice = st.selectbox(
    "请选择一个模型:",
    ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]
)

# 根据选择加载模型
if model_choice == "Linear Regression":
    model = load("linear_regression_model.joblib")
elif model_choice == "Decision Tree":
    model = load("decision_tree_model.joblib")
elif model_choice == "Random Forest":
    model = load("random_forest_model.joblib")
elif model_choice == "XGBoost":
    model = load("xgboost_model.joblib")

# 用户输入
area = st.number_input("请输入房屋面积 (平方米):", min_value=10, max_value=1000, step=10)
bedrooms = st.slider("卧室数量:", 1, 10, 3)

# 构造输入
input_array = np.array([area, bedrooms]).reshape(1, -1)

# 预测
if st.button("预测房价"):
    predicted_price = model.predict(input_array)
    st.success(f"预测的房价是: {predicted_price[0]:.2f} 万元")
