import streamlit as st
import pandas as pd
import joblib

st.title("Dry Bean Classifier")
st.write("Upload test data, pick a model, and see how it performs.")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree":        joblib.load("model/decision_tree.pkl"),
        "kNN":                  joblib.load("model/knn.pkl"),
        "Naive Bayes":          joblib.load("model/naive_bayes.pkl"),
        "Random Forest":        joblib.load("model/random_forest.pkl"),
    }
    return scaler, models

scaler, models = load_artifacts()
st.success(f"Loaded {len(models)} models and the scaler.")

st.header("1. Upload test data")
uploaded = st.file_uploader("Upload your test_data.csv", type="csv")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write(f"Uploaded data shape: {data.shape[0]} rows, {data.shape[1]} columns")
    st.dataframe(data.head())

    st.header("2. Choose a model")
    model_name = st.selectbox("Select a model", list(models.keys()))

    X_new = data.drop(columns=["Class"])
    y_true = data["Class"]

    X_scaled = scaler.transform(X_new)

    chosen_model = models[model_name]
    y_pred = chosen_model.predict(X_scaled)

    st.success(f"Ran {model_name} on {len(y_true)} rows.")

    # ---- 3. Evaluation metrics ----
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, matthews_corrcoef)

    st.header("3. Evaluation metrics")
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall":    recall_score(y_true, y_pred, average="macro"),
        "F1":        f1_score(y_true, y_pred, average="macro"),
        "MCC":       matthews_corrcoef(y_true, y_pred),
    }
    metrics_df = pd.DataFrame(metrics, index=[model_name]).round(4)
    st.dataframe(metrics_df)

    # ---- 4. Confusion matrix ----
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("4. Confusion matrix")
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    st.pyplot(fig)

else:
    st.info("Waiting for a CSV file...")
