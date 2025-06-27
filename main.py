import streamlit as st
import pandas as pd
import shap
import pickle
from model import explainer, threshold  # model.py에서 정의된 explainer와 threshold 사용

# 모델 로드
with open("xgb_sleep_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 수면장애 위험 예측")

# 직업 선택
occupation_dict = {
    "Accountant": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3, "Manager": 4,
    "Nurse": 5, "Sales Representative": 6, "Salesperson": 7,
    "Scientist": 8, "Software Engineer": 9, "Teacher": 10
}
occupation_label = st.selectbox("직업을 선택하세요", list(occupation_dict.keys()))
occupation = occupation_dict[occupation_label]

# BMI 선택
bmi_dict = {
    "저체중 (Underweight)": 0, "정상체중 (Normal)": 1,
    "과체중 (Overweight)": 2, "비만 (Obese)": 3
}
bmi_label = st.selectbox("BMI 범주를 선택하세요", list(bmi_dict.keys()))
bmi_category = bmi_dict[bmi_label]

# 걸음 수 & 수면 시간
daily_steps = st.slider("하루 평균 걸음 수", 0, 20000, 4000, step=100)
sleep_duration = st.slider("하루 평균 수면 시간 (시간)", 0, 12, 7, step=1)

# 스트레스 레벨 (0~11)
stress_level = st.slider("스트레스 수준 (0 = 전혀 없음, 11 = 매우 높음)", 0, 11, 5, step=1)

# 입력값을 데이터프레임으로 구성
user_input = {
    'BMI Category': bmi_category,
    'Occupation': occupation,
    'Daily Steps': daily_steps,
    'Sleep Duration': sleep_duration,
    'Stress Level': stress_level
}
user_df = pd.DataFrame([user_input])

# 사용자 입력 확인
st.subheader("📋 입력된 정보")
st.write(user_df)

# 예측 실행 버튼
if st.button("🧪 위험도 예측하기"):
    # 예측 확률
    proba = model.predict_proba(user_df)[0, 1]

    # SHAP 점수
    user_shap = explainer.shap_values(user_df)
    shap_score = user_shap.sum()

    # 위험 수준 판단 (proba 기준)
    if proba >= 0.8:
        risk_by_proba = 'High'
    elif proba >= 0.6:
        risk_by_proba = 'Medium'
    else:
        risk_by_proba = 'Low'

    # 위험 수준 판단 (SHAP 기준)
    risk_by_shap = 'High' if shap_score >= threshold else 'Low'

    # 최종 판단
    if risk_by_proba == 'High' or risk_by_shap == 'High':
        final_risk = 'High'
    elif risk_by_proba == 'Medium':
        final_risk = 'Medium'
    else:
        final_risk = 'Low'

    # 결과 출력
    st.subheader("📊 예측 결과")
    st.write(f"🧠 예측 확률: **{proba:.3f}**")
    st.write(f"🔥 SHAP 위험 점수: **{shap_score:.3f}** (기준값: {threshold:.3f})")
    st.markdown(f"## ✅ 최종 위험군: **:red[{final_risk}]**" if final_risk == 'High' else
                f"## ⚠️ 최종 위험군: **:orange[{final_risk}]**" if final_risk == 'Medium' else
                f"## 🟢 최종 위험군: **{final_risk}**")
