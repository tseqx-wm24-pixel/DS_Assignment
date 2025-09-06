import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="房价预测系统", page_icon="🏠")

st.title("🏠 房价预测系统")

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

# 选择模型
model_choice = st.selectbox("请选择一个模型:", list(models.keys()))

# 输入特征（假设有面积和卧室数两个特征）
area = st.number_input("请输入房屋面积 (平方米):", min_value=10, max_value=1000, step=10)
bedrooms = st.slider("卧室数量:", 1, 10, 3)

if st.button("预测房价"):
    model = models[model_choice]
    pred = model.predict(np.array([[area, bedrooms]]))[0]
    st.success(f"✅ 使用 {model_choice} 预测的房价为: {pred:.2f} 万元")
