import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Load model and preprocessing objects ---
@st.cache_resource
def load_models():
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler.joblib')
    imputer = joblib.load('imputer.joblib')
    encoder = joblib.load('encoder.joblib')
    pca = joblib.load('pca_connectome.joblib')
    
    with open('adhd_lr_threshold.json', 'r') as f:
        threshold = json.load(f)['threshold']
    
    with open('quant_cols.json', 'r') as f:
        quant_cols = json.load(f)
    with open('cat_cols.json', 'r') as f:
        cat_cols = json.load(f)
    with open('conn_cols.json', 'r') as f:
        conn_cols = json.load(f)
    
    return model, scaler, imputer, encoder, pca, threshold, quant_cols, cat_cols, conn_cols

model, scaler, imputer, encoder, pca, threshold, quant_cols, cat_cols, conn_cols = load_models()

# --- Streamlit UI ---
st.title("ADHD Prediction App")

with st.expander("ℹ️ Data Dictionary"):
    try:
        data_dict_df = pd.read_excel("Data Dictionary/Data Dictionary (1) (1).xlsx")
        st.dataframe(data_dict_df)
    except:
        st.write("Data dictionary not found")

st.write("Please answer the following questions:")

# --- User Inputs ---
user_quant = {}
for col in quant_cols:
    user_quant[col] = st.number_input(f"{col}", value=0.0, key=f"quant_{col}")

user_cat = {}
for col in cat_cols:
    user_cat[col] = st.number_input(f"{col}", value=0.0, key=f"cat_{col}")

st.write("Upload your connectome data as a CSV file:")
conn_file = st.file_uploader("Upload connectome CSV", type=["csv"])

if conn_file is not None:
    try:
        conn_df = pd.read_csv(conn_file)
        user_conn = {col: conn_df.iloc[0][col] if col in conn_df.columns else 0.0 for col in conn_cols}
        st.success(f"Loaded connectome data with {len(conn_df.columns)} features")
    except Exception as e:
        st.error(f"Error loading connectome file: {e}")
        user_conn = {col: 0.0 for col in conn_cols}
else:
    user_conn = {col: 0.0 for col in conn_cols}
    st.info("Using default values (0.0) for connectome data")

# --- Prepare DataFrame for preprocessing ---
input_df = pd.DataFrame([{**user_quant, **user_cat, **user_conn}])

# --- Prediction ---
if st.button("Predict ADHD"):
    try:
        # Ensure all expected columns are present
        input_df = input_df.reindex(columns=quant_cols + cat_cols + conn_cols, fill_value=0.0)
        
        # 1. Impute missing values (on quant + cat columns)
        input_df[quant_cols + cat_cols] = imputer.transform(input_df[quant_cols + cat_cols])
        
        # 2. Scale quantitative columns
        input_df[quant_cols] = scaler.transform(input_df[quant_cols])
        
        # 3. Encode categorical columns
        encoded = encoder.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)
        input_df = input_df.drop(columns=cat_cols)
        input_df = pd.concat([input_df, encoded_df], axis=1)
        
        # 4. PCA for connectome - FIXED COLUMN NAMES
        conn_pca = pca.transform(input_df[conn_cols])
        pca_cols = [f'conn_pca_{i+1}' for i in range(conn_pca.shape[1])]  # Match training
        conn_pca_df = pd.DataFrame(conn_pca, columns=pca_cols, index=input_df.index)
        input_df = input_df.drop(columns=conn_cols)
        input_df = pd.concat([input_df, conn_pca_df], axis=1)
        
        # 5. Make prediction
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= threshold)
        
        st.subheader("Prediction Result")
        st.write(f"Probability of ADHD: {proba:.2%}")
        st.write(f"Threshold used: {threshold}")
        
        if prediction == 1:
            st.markdown("**Likely ADHD** 🔴")
        else:
            st.markdown("**Unlikely ADHD** 🟢")
        
        st.info("⚠️ This prediction is for informational purposes only and not a medical diagnosis.")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please check that all inputs are valid and try again.")