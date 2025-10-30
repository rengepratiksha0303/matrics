# ============================================================
# 🧠 Kraljic Matrix Procurement Classification Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# Streamlit App Layout
# ============================================================

st.set_page_config(page_title="Kraljic Matrix Classifier", layout="wide")
st.title("🧠 Kraljic Matrix Procurement Classification")
st.write("Upload your dataset to train models and predict procurement categories.")

# ============================================================
# Upload Training File
# ============================================================

train_file = st.file_uploader("📂 Upload Training CSV (realistic_kraljic_dataset.csv)", type=["csv"])

if train_file:
    df = pd.read_csv(train_file)
    st.success("✅ Training dataset loaded successfully!")
    st.dataframe(df.head())

    # ============================================================
    # Basic Info
    # ============================================================
    with st.expander("📊 Dataset Overview"):
        st.write(df.describe())
        st.write(df.info())

    # ============================================================
    # Drop String Columns Except Target
    # ============================================================
    if 'Kraljic_Category' not in df.columns:
        st.error("❌ Column 'Kraljic_Category' not found in dataset.")
    else:
        string_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'Kraljic_Category' in string_cols:
            string_cols.remove('Kraljic_Category')
        df = df.drop(columns=string_cols)
        st.write(f"🧹 Dropped string columns: {string_cols}")

        # ============================================================
        # Encode Target Column
        # ============================================================
        le_target = LabelEncoder()
        df['Kraljic_Category'] = le_target.fit_transform(df['Kraljic_Category'])

        # ============================================================
        # Split Data
        # ============================================================
        X = df.drop('Kraljic_Category', axis=1)
        y = df['Kraljic_Category']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # ============================================================
        # Train Multiple Models
        # ============================================================
        st.subheader("⚙ Training Models...")

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel='rbf', probability=True),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = {}

        progress = st.progress(0)
        for i, (name, model) in enumerate(models.items(), 1):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            st.write(f"✅ {name} Accuracy: {acc:.4f}")
            progress.progress(i / len(models))

        # ============================================================
        # Model Comparison Chart
        # ============================================================
        st.subheader("📈 Model Accuracy Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax)
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # ============================================================
        # Best Model
        # ============================================================
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        st.success(f"🏆 Best Model: {best_model_name} ({results[best_model_name]:.2f} accuracy)")

        # ============================================================
        # Upload Test Data for Prediction
        # ============================================================
        st.subheader("🔮 Upload Test Data for Prediction")
        test_file = st.file_uploader("📂 Upload Test CSV", type=["csv"], key="test")

        if test_file:
            test_df = pd.read_csv(test_file)
            st.write("✅ Test data loaded!")
            st.dataframe(test_df.head())

            # Drop same string columns
            test_df = test_df.drop(columns=string_cols, errors='ignore')

            # Scale numeric features
            test_scaled = scaler.transform(test_df)

            # Predict
            predictions = best_model.predict(test_scaled)
            test_df['Predicted_Category'] = le_target.inverse_transform(predictions)

            st.subheader("🎯 Prediction Results")
            st.dataframe(test_df[['Predicted_Category']])

            # Download button
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="Kraljic_Predictions.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Please upload your training dataset to get started.")
