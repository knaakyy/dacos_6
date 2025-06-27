import pandas as pd
import shap
import pickle

# 데이터 불러오기
df = pd.read_csv("df_preprocessed.csv")

# 사용 feature: 혈압 제거, 스트레스 레벨 포함
X = df[['BMI Category', 'Occupation', 'Daily Steps', 'Sleep Duration', 'Stress Level']]

# 모델 로드
with open("xgb_sleep_model.pkl", "rb") as f:
    model = pickle.load(f)

# 예측 확률
df['Disorder_Prob'] = model.predict_proba(X)[:, 1]
df['Risk_by_Proba'] = df['Disorder_Prob'].apply(
    lambda x: 'High' if x >= 0.8 else 'Medium' if x >= 0.6 else 'Low'
)

# SHAP 값 계산
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

df['SHAP_Risk_Score'] = shap_values.sum(axis=1)
threshold = df['SHAP_Risk_Score'].quantile(0.7)

df['Risk_by_SHAP'] = df['SHAP_Risk_Score'].apply(
    lambda x: 'High' if x >= threshold else 'Low'
)

df['Final_Risk_Level'] = df.apply(
    lambda row: 'High' if row['Risk_by_Proba'] == 'High' and row['Risk_by_SHAP'] == 'High' else 'Not High',
    axis=1
)
