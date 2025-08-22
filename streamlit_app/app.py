# ================================================
# 📌 Streamlit App - Adaptive Credit Scoring (Dark Mode + Explainability)
# ================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from treeinterpreter import treeinterpreter as ti
from datetime import datetime

# ----------------------------
# 🎨 Custom Dark CSS Styling
# ----------------------------
dark_style = """
<style>
    body {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stMarkdown, .stDataFrame, .stTable, .stTextInput, .stFileUploader {
        color: #e0e0e0 !important;
    }
    h1, h2, h3, h4 {
        color: #f5c542 !important;
    }
    .css-18e3th9 {
        background-color: #1e2228;
        border-radius: 12px;
        padding: 12px;
    }
    .stButton>button {
        background-color: #f54291;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #42a5f5;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)

# ----------------------------
# ⚙️ Config
# ----------------------------
MODEL_PATH = "decision_tree_credit_model.pkl"

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

# ----------------------------
# 📌 Explainability Function
# ----------------------------
def explain_dataset(model, df):
    feature_cols = [c for c in df.columns if c not in ['issuer_id','last_updated','creditworthiness_score']]
    X_test = df[feature_cols].copy()

    # Align features with training model
    train_features = model.feature_names_in_.tolist()
    for col in train_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_features]

    # TreeInterpreter
    prediction, bias, contributions = ti.predict(model, X_test)

    contrib_df = pd.DataFrame(contributions, columns=[f"{c}_contrib" for c in train_features])
    df_explain = X_test.reset_index(drop=True).copy()
    df_explain = pd.concat([df_explain, contrib_df], axis=1)
    df_explain['predicted_score'] = prediction

    # Add issuer_id & last_updated
    if 'issuer_id' in df.columns:
        df_explain['issuer_id'] = df['issuer_id']
    else:
        df_explain['issuer_id'] = np.arange(len(df_explain))
    if 'last_updated' in df.columns:
        df_explain['last_updated'] = pd.to_datetime(df['last_updated'])
    else:
        df_explain['last_updated'] = pd.Timestamp.now()

    # Trends
    df_explain = df_explain.sort_values(by=['issuer_id','last_updated'])
    df_explain['short_term_trend'] = df_explain.groupby('issuer_id')['predicted_score'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df_explain['long_term_trend'] = df_explain.groupby('issuer_id')['predicted_score'].transform(lambda x: x.rolling(10, min_periods=1).mean())

    # Explanation text
    def explain_row(row):
        contrib_cols = [c for c in df_explain.columns if c.endswith("_contrib")]
        top_features = row[contrib_cols].sort_values(key=lambda x: abs(x), ascending=False).head(3)
        reasoning = "Top contributing features:\n"
        for f, val in top_features.items():
            feature_name = f.replace("_contrib", "")
            feature_value = row[feature_name] if feature_name in row else "NA"
            reasoning += f"- {feature_name}: value={feature_value:.3f}, contribution={val:.3f}\n"
        if 'news_pos_ratio' in row and row['news_pos_ratio'] > 0.5:
            reasoning += "- Positive news sentiment is high, boosting score.\n"
        if 'news_neg_ratio' in row and row['news_neg_ratio'] > 0.5:
            reasoning += "- Negative news sentiment is high, reducing score.\n"
        reasoning += f"- Short-term trend: {row['short_term_trend']:.2f}, Long-term trend: {row['long_term_trend']:.2f}\n"
        return reasoning

    df_explain['explanation'] = df_explain.apply(explain_row, axis=1)

    return df_explain[['issuer_id','predicted_score','short_term_trend','long_term_trend','explanation']]

# ----------------------------
# 🚀 Streamlit App
# ----------------------------
st.title("🧠Adaptive Credit Scoring with Explainability")
st.markdown("Upload a CSV file and get **detailed explanations** for each issuer's credit score.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Preview of Uploaded Data")
    st.dataframe(df.head())

    model = load_model()
    if model is None:
        st.error("⚠️ No saved model found. Please train and save the model first.")
    else:
        if st.button("⚡ Generate Explanations"):
            explained_df = explain_dataset(model, df)

            st.subheader("📝 Credit Scoring Explanations")
            st.dataframe(explained_df)

            st.download_button(
                label="⬇️ Download Explanations (CSV)",
                data=explained_df.to_csv(index=False),
                file_name="credit_score_explanations.csv",
                mime="text/csv"
            )
