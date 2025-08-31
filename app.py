import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------- Page Setup ----------
st.set_page_config(page_title="Healthcare Dashboard", layout="wide")
st.title("📊 Healthcare Demand and Model Results Dashboard")
st.caption("Assessment 4 – Task One")

# ---------- Robust CSV Loader ----------
def safe_read_csv(name):
    p = Path(name)
    if not p.exists():
        st.error(f"❌ Missing required file: {name}. Please export it from your notebook first.")
        st.stop()
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep=";")   # try semicolon-delimited
        except Exception:
            try:
                return pd.read_csv(p, sep="\t")  # try tab-delimited
            except Exception as e:
                st.error(f"❌ Failed to load {name}: {e}")
                st.stop()

# ---------- Load Data ----------
df = safe_read_csv("df.csv")
daily = safe_read_csv("daily.csv")
results_df = safe_read_csv("results_df.csv")
gender_counts = safe_read_csv("gender_counts.csv")

# ---------- Train Models ----------
target_col = df.columns[-1]   # assume last column is target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == "object" else col)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)

# Calculate accuracy
acc_lr = accuracy_score(y_test, log_reg.predict(X_test))
acc_rf = accuracy_score(y_test, rf.predict(X_test))

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📈 Data & Insights", "🤖 Model Results", "🧠 Manual Prediction"])

# ------------- Tab 1: Data & Insights -------------
with tab1:
    st.subheader("Quick Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{len(df)}")
    col2.metric("Columns", f"{df.shape[1]}")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Daily Hospital Demand")
    st.dataframe(daily.head())

    st.markdown("### Gender-Based Admissions")
    st.dataframe(gender_counts)

    # Plotting Daily Demand Over Time
    st.markdown("### Daily Demand Over Time")
    fig, ax = plt.subplots()
    ax.plot(daily["Date of Admission"], daily["y_hosp"])
    ax.set_xlabel("Date of Admission")
    ax.set_ylabel("Hospital Demand")
    ax.set_title("Daily Hospital Demand")
    st.pyplot(fig)

    # Plotting Admissions by Gender
    st.markdown("### Admissions by Gender")
    fig2, ax2 = plt.subplots()
    ax2.bar(gender_counts["Gender"], gender_counts["Admissions"])
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Admissions")
    ax2.set_title("Admissions by Gender")
    st.pyplot(fig2)

# ------------- Tab 2: Model Results -------------
with tab2:
    st.subheader("Model Comparison")

    # Show the model accuracy results
    st.markdown("### Model Accuracy Comparison")
    st.dataframe(results_df.sort_values("Accuracy", ascending=False), use_container_width=True)

    # Show summary accuracy
    st.markdown("### 🔍 Test Accuracy")
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression", f"{acc_lr:.2%}")
    col2.metric("Random Forest", f"{acc_rf:.2%}")

    # Plot model comparison
    st.markdown("### Logistic Regression vs Random Forest")
    fig3, ax3 = plt.subplots()
    ax3.bar(results_df["Model"], results_df["Accuracy"])
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Model Accuracy Comparison")
    st.pyplot(fig3)

# ------------- Tab 3: Manual Prediction -------------
with tab3:
    st.subheader("Enter Patient Features to Predict Outcome")

    user_input = {}
    for col in X.columns:
        if df[col].dtype == "object":
            options = df[col].unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            user_input[col] = st.slider(f"{col}", min_val, max_val, float(df[col].median()))

    input_df = pd.DataFrame([user_input])

    # Encode categorical inputs to match training
    for col in input_df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            encoder.fit(df[col])
            input_df[col] = encoder.transform(input_df[col])

    if st.button("Predict Outcome"):
        pred_lr = log_reg.predict(input_df)[0]
        pred_rf = rf.predict(input_df)[0]

        st.success(f"Logistic Regression Prediction: {pred_lr}")
        st.info(f"Random Forest Prediction: {pred_rf}")
