import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

# ==================================================
# 2. APPLICATION HEADER
# ==================================================

st.title("🔬 Breast Cancer Classification")
st.subheader("Machine Learning Model Comparison")

st.markdown("""
This application demonstrates five machine-learning classification
algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

Upload your test dataset, select a trained model and evaluate
its prediction performance.
""")

st.info(
    "Class mapping: 0 = Malignant | 1 = Benign. "
    "This is an academic demonstration, not a clinical diagnostic tool."
)

st.divider()

# ==================================================
# 3. MODEL CONFIGURATION
# ==================================================

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours (KNN)": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "svm.joblib"
}

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# ==================================================
# 4. DATASET UPLOAD
# ==================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.write("Dataset Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        st.write(
            f"Number of records: **{uploaded_df.shape[0]}** | "
            f"Number of columns: **{uploaded_df.shape[1]}**"
        )

        # ==================================================
        # 5. MODEL SELECTION
        # ==================================================

        st.header("2. Select Machine Learning Model")

        selected_model = st.selectbox(
            "Choose a trained model",
            list(MODEL_FILES.keys())
        )

        # ==================================================
        # 6. RUN PREDICTION
        # ==================================================

        if st.button("Run Prediction", type="primary"):

            if "diagnosis" not in uploaded_df.columns:
                st.error(
                    "The uploaded CSV must contain the diagnosis column."
                )
                st.stop()

            if uploaded_df.empty:
                st.error("The uploaded CSV contains no records.")
                st.stop()

            y_actual = uploaded_df["diagnosis"]

            if y_actual.isna().any():
                st.error("The diagnosis column contains missing values.")
                st.stop()

            if not set(y_actual.unique()).issubset({0, 1}):
                st.error(
                    "The diagnosis column must contain only 0 and 1."
                )
                st.stop()

            X_uploaded = uploaded_df.drop(columns=["diagnosis"])

            model_path = os.path.join(
                MODEL_DIR,
                MODEL_FILES[selected_model]
            )

            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                st.stop()

            with st.spinner("Loading model and generating predictions..."):

                model = load_model(model_path)

                expected_features = list(model.feature_names_in_)

                if set(X_uploaded.columns) != set(expected_features):
                    st.error(
                        "Incorrect input columns. Please upload the "
                        "assignment test_data.csv with all 30 features."
                    )
                    st.stop()

                # Keep the same feature order used during training.
                X_uploaded = X_uploaded[expected_features]

                # Generate predictions.
                y_pred = model.predict(X_uploaded)

                # Find probability corresponding to class 1 (Benign).
                class_index = list(model.classes_).index(1)

                y_probability = model.predict_proba(
                    X_uploaded
                )[:, class_index]

            # ==================================================
            # 7. CALCULATE PERFORMANCE METRICS
            # ==================================================

            accuracy = accuracy_score(y_actual, y_pred)

            auc = roc_auc_score(
                y_actual,
                y_probability
            )

            precision = precision_score(
                y_actual, y_pred, zero_division=0
            )

            recall = recall_score(
                y_actual, y_pred, zero_division=0
            )

            f1 = f1_score(
                y_actual, y_pred, zero_division=0
            )

            mcc = matthews_corrcoef(y_actual, y_pred)

            # ==================================================
            # 8. DISPLAY PERFORMANCE METRICS
            # ==================================================

            st.divider()
            st.header("3. Model Performance")

            st.write(f"Selected model: **{selected_model}**")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.2%}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.2%}")
            col5.metric("F1 Score", f"{f1:.2%}")
            col6.metric("MCC", f"{mcc:.4f}")

            # ==================================================
            # 9. CONFUSION MATRIX
            # ==================================================

            st.divider()
            st.header("4. Confusion Matrix")

            cm = confusion_matrix(
                y_actual,
                y_pred,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Malignant", "Actual Benign"],
                columns=["Predicted Malignant", "Predicted Benign"]
            )

            st.dataframe(cm_df, use_container_width=True)

            # ==================================================
            # 10. CLASSIFICATION REPORT
            # ==================================================

            st.header("5. Classification Report")

            report = classification_report(
                y_actual,
                y_pred,
                labels=[0, 1],
                target_names=["Malignant", "Benign"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            # ==================================================
            # 11. PREDICTION RESULTS
            # ==================================================

            st.header("6. Prediction Results")

            prediction_df = uploaded_df.copy()

            prediction_df["Predicted Diagnosis"] = y_pred

            prediction_df["Predicted Label"] = (
                prediction_df["Predicted Diagnosis"]
                .map({
                    0: "Malignant",
                    1: "Benign"
                })
            )

            prediction_df["Probability of Benign"] = y_probability

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

            csv_output = prediction_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.warning("Please upload your test_data.csv file to begin.")

# ==================================================
# 12. FOOTER
# ==================================================

st.divider()

st.caption(
    "BITS Pilani WILP | M.Tech AIML | Machine Learning Assignment 2"
)
