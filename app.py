import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Machine Learning Classification Evaluation App")

st.subheader("Download Sample Test Data")

# Open file in binary mode
with open("data/UCI_Credit_CardTest.csv", "rb") as file:
    st.download_button(
        label="Download Test CSV",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )

TARGET_COL = "default.payment.next.month"

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

models = load_models()

def load_scaler():
    return joblib.load("model/standard_scaler.pkl")

scaler = load_scaler()

SCALING_MODELS = [
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
]

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    if TARGET_COL not in data.columns:
        st.error(f"Target column '{TARGET_COL}' not found in uploaded file.")
        st.stop()

    X_test = data.drop(columns=[TARGET_COL])
    y_test = data[TARGET_COL]
    X_eval = X_test.copy()

   
    # ----------------------------
    # Model selection
    # ----------------------------
    model_name = st.selectbox("Select a model", list(models.keys()))
    model = models[model_name]
    if model_name in SCALING_MODELS:
        X_eval = scaler.transform(X_eval)

    if st.button("Evaluate Model"):
        y_pred = model.predict(X_eval)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = "N/A"

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"{model_name} Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 4))
        col1.metric("Precision", round(prec, 4))

        col2.metric("Recall", round(rec, 4))
        col2.metric("F1 Score", round(f1, 4))

        col3.metric("ROC-AUC", round(roc_auc, 4) if roc_auc != "N/A" else "N/A")
        col3.metric("MCC", round(mcc, 4))

        # ----------------------------
        # Confusion Matrix
        # ----------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ----------------------------
        # Classification Report
        # ----------------------------
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())