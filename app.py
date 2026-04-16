# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Menstrual Cycle Prediction", layout="wide")
st.title("Menstrual Cycle Regularity Prediction Dashboard")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("menstrual_cycle_dataset_with_factors_01.csv")
st.success("Dataset Loaded Successfully")

# ===============================
# TARGET CREATION
# ===============================
df["Regularity"] = df["Cycle Length"].apply(lambda x: 1 if 21 <= x <= 35 else 0)

# ===============================
# ENCODING
# ===============================
encoders = {}
for col in ["Exercise Frequency", "Diet", "Symptoms"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ===============================
# FEATURES & TARGET
# ===============================
X = df[
    ["Age", "BMI", "Stress Level", "Exercise Frequency",
     "Sleep Hours", "Diet", "Cycle Length", "Period Length", "Symptoms"]
]
y = df["Regularity"]

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ===============================
# SCALE (ANN)
# ===============================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===============================
# MODELS
# ===============================
xgb = XGBClassifier(
    n_estimators=120, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=1.0,
    eval_metric="logloss", random_state=42
)

ann = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    max_iter=500,
    alpha=0.01,
    random_state=42
)

xgb.fit(X_train, y_train)
ann.fit(X_train_s, y_train)
st.success("Model Trained Successfully")

# ===============================
# HYBRID PREDICTION
# ===============================
xgb_prob = xgb.predict_proba(X_test)[:, 1]
ann_prob = ann.predict_proba(X_test_s)[:, 1]
hybrid_pred = ((xgb_prob + ann_prob) / 2 > 0.5).astype(int)

accuracy = accuracy_score(y_test, hybrid_pred)
st.subheader("Model Accuracy")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# ===============================
# PERSONAL PREDICTION
# ===============================
st.subheader("Personal Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 15, 50, 25)
    bmi = st.number_input("BMI", 15.0, 40.0, 22.0)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

with col2:
    exercise = st.selectbox("Exercise Frequency", encoders["Exercise Frequency"].classes_)
    sleep = st.number_input("Sleep Hours per Night", 3.0, 10.0, 7.0)
    diet = st.selectbox("Diet", encoders["Diet"].classes_)

with col3:
    cycle_len = st.number_input("Cycle Length (days)", 15, 45, 28)
    period_len = st.number_input("Period Length (days)", 2, 10, 5)
    symptoms = st.selectbox("Symptoms", encoders["Symptoms"].classes_)

if st.button("Get Prediction"):
    sample = pd.DataFrame([[
        age, bmi, stress,
        encoders["Exercise Frequency"].transform([exercise])[0],
        sleep,
        encoders["Diet"].transform([diet])[0],
        cycle_len, period_len,
        encoders["Symptoms"].transform([symptoms])[0]
    ]], columns=X.columns)

    p1 = xgb.predict_proba(sample)[:, 1]
    p2 = ann.predict_proba(scaler.transform(sample))[:, 1]
    final = "Regular" if ((p1 + p2) / 2 > 0.5) else "Irregular"

    st.success(f"Prediction Result: {final}")

# ===============================
# INDIVIDUAL PREDICTIONS TABLE
# ===============================
st.subheader("Individual Predictions")

df_pred = df.copy()
df_pred["Prediction"] = [
    "Regular" if p == 1 else "Irregular"
    for p in ((xgb.predict_proba(X)[:,1] +
               ann.predict_proba(scaler.transform(X))[:,1]) / 2 > 0.5)
]

df_pred["Prediction"] = ["Woman " + str(i+1) + " → " + p
                          for i, p in enumerate(df_pred["Prediction"])]

st.dataframe(df_pred[["Prediction"]])

# ===============================
# TOP REASONS TABLE
# ===============================
st.subheader("Top Reasons for Irregular Periods")

importance = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

st.table(importance.head(5))

# ===============================
# REGULARITY DISTRIBUTION
# ===============================
st.subheader("Regularity Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(x=y, ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Irregular", "Regular"])
st.pyplot(fig1)

# ===============================
# FACTORS CONTRIBUTING BAR CHART
# ===============================
st.subheader("Factors Contributing to Irregular Periods")

fig2, ax2 = plt.subplots()
sns.barplot(x=importance.values, y=importance.index, ax=ax2)
st.pyplot(fig2)

# ===============================
# CONFUSION MATRIX
# ===============================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, hybrid_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Irregular", "Regular"],
            yticklabels=["Irregular", "Regular"],
            cmap="Blues")
st.pyplot(fig3)

