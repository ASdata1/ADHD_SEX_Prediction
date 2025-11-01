import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Load model and preprocessing objects ---
@st.cache_resource
def load_models():
    """
    Load all model artifacts needed for ADHD prediction.
    
    Returns:
    --------
    tuple
        All loaded artifacts needed for prediction
    """
    try:
        # Load preprocessing objects
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        encoder = joblib.load('encoder.joblib')
        pca = joblib.load('pca_connectome.joblib')
        
        # Load model report and extract info
        with open('adhd_model_report.json', 'r') as f:
            model_report = json.load(f)
        
        model_info = model_report['model_info']
        model_name = model_info['name']  # "Logistic Regression"
        optimal_threshold = model_info['optimal_threshold'] 
        
        # Load the actual trained model
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        trained_model = joblib.load(model_filename)
        
        # Load feature column definitions
        with open('quant_cols.json', 'r') as f:
            quant_cols = json.load(f)
        with open('cat_cols.json', 'r') as f:
            cat_cols = json.load(f)
        with open('conn_cols.json', 'r') as f:
            conn_cols = json.load(f)
        
        return (scaler, imputer, encoder, pca, trained_model, 
                optimal_threshold, quant_cols, cat_cols, conn_cols)
        
    except FileNotFoundError as e:
        st.error(f"Required model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load all artifacts
(scaler, imputer, encoder, pca, model,  # ✅ FIXED: Use 'model' not 'trained_model'
 threshold, quant_cols, cat_cols, conn_cols) = load_models()  # ✅ FIXED: Use 'threshold' not 'optimal_threshold'

# --- Streamlit UI ---
st.title("🧠 ADHD Prediction App")
st.markdown("---")

# Model information display
st.sidebar.header("📊 Model Information")
st.sidebar.info(f"""
**Algorithm:** Logistic Regression  
**Features:** {len(quant_cols + cat_cols + conn_cols)} total  
**Quantitative:** {len(quant_cols)}  
**Categorical:** {len(cat_cols)}  
**Connectome:** {len(conn_cols)}  
**Optimal Threshold:** {threshold:.3f}
""")

with st.expander("ℹ️ Data Dictionary"):
    try:
        data_dict_df = pd.read_excel("C:\\Users\\04ama\\OneDrive\\chemistry\\ADHD_SEX_Prediction\\Data Dictionary\\Data Dictionary (1).xlsx")
        st.dataframe(data_dict_df)
    except Exception as e:  # ✅ FIXED: Specific exception handling
        st.warning(f"Data dictionary not found: {e}")

st.write("Please provide the following information:")

# --- User Inputs with better organization ---
st.subheader("📋 Questionnaire Data")

# Create columns for better layout
col1, col2 = st.columns(2)

# Split quantitative features between columns
mid_point = len(quant_cols) // 2

user_quant = {}
with col1:
    st.write("**Quantitative Features (Part 1):**")
    for col in quant_cols[:mid_point]:
        user_quant[col] = st.number_input(f"{col}", value=0.0, key=f"quant_{col}")

with col2:
    st.write("**Quantitative Features (Part 2):**")
    for col in quant_cols[mid_point:]:
        user_quant[col] = st.number_input(f"{col}", value=0.0, key=f"quant_{col}")

st.subheader("Categorical Data")
user_cat = {}
for col in cat_cols:
    user_cat[col] = st.selectbox(f"{col}", [0.0, 1.0], key=f"cat_{col}")  # ✅ FIXED: Use selectbox for binary categories

st.subheader("Connectome Data")
st.write("Upload your connectome data as a CSV file:")
conn_file = st.file_uploader("Upload connectome CSV", type=["csv"])

if conn_file is not None:
    try:
        conn_df = pd.read_csv(conn_file)
        
        # ✅ FIXED: Better error handling for missing columns
        user_conn = {}
        missing_cols = []
        for col in conn_cols:
            if col in conn_df.columns:
                user_conn[col] = conn_df.iloc[0][col]
            else:
                user_conn[col] = 0.0
                missing_cols.append(col)
        
        if missing_cols:
            st.warning(f"Missing {len(missing_cols)} connectome features. Using default values.")
        else:
            st.success(f"✅ Loaded connectome data with {len(conn_df.columns)} features")
            
    except Exception as e:
        st.error(f"Error loading connectome file: {e}")
        user_conn = {col: 0.0 for col in conn_cols}
else:
    user_conn = {col: 0.0 for col in conn_cols}
    st.info("ℹ️ Using default values (0.0) for connectome data")

# --- Prepare DataFrame for preprocessing ---
input_df = pd.DataFrame([{**user_quant, **user_cat, **user_conn}])

# --- Prediction ---
st.markdown("---")
if st.button("🎯 Predict ADHD", type="primary"):
    try:
        # Display processing steps
        with st.spinner("Processing data"):
            
            # Ensure all expected columns are present
            input_df = input_df.reindex(columns=quant_cols + cat_cols + conn_cols, fill_value=0.0)
            
            # 1. Scale quantitative columns
            if len(quant_cols) > 0:  # FIXED: Check if columns exist
                input_df[quant_cols] = scaler.transform(input_df[quant_cols])
            
            # 2. Impute missing values (on quant + cat columns)
            if len(quant_cols + cat_cols) > 0:  # FIXED: Check if columns exist
                input_df[quant_cols + cat_cols] = imputer.transform(input_df[quant_cols + cat_cols])
            
            # 3. Encode categorical columns
            if len(cat_cols) > 0:  # FIXED: Check if columns exist
                encoded = encoder.transform(input_df[cat_cols])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)
                input_df = input_df.drop(columns=cat_cols)
                input_df = pd.concat([input_df, encoded_df], axis=1)
            
            # 4. PCA for connectome
            if len(conn_cols) > 0:  # FIXED: Check if columns exist
                conn_pca = pca.transform(input_df[conn_cols])
                pca_cols = [f'conn_pca_{i+1}' for i in range(conn_pca.shape[1])]
                conn_pca_df = pd.DataFrame(conn_pca, columns=pca_cols, index=input_df.index)
                input_df = input_df.drop(columns=conn_cols)
                input_df = pd.concat([input_df, conn_pca_df], axis=1)
            
            # 5. Make prediction
            proba = model.predict_proba(input_df)[0][1]
            prediction = int(proba >= threshold)
        
        # Display results with better formatting
        st.success("✅ Prediction completed!")
        
        # Create results section
        st.subheader("🎯 Prediction Results")
        
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ADHD Probability", f"{proba:.1%}")
        
        with col2:
            st.metric("Threshold Used", f"{threshold:.3f}")
        
        with col3:
            confidence = max(proba, 1-proba)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Main prediction result
        if prediction == 1:
            st.error("🔴 **Likely ADHD**")
            st.write(f"The model predicts this individual **likely has ADHD** with {proba:.1%} probability.")
        else:
            st.success("🟢 **Unlikely ADHD**")
            st.write(f"The model predicts this individual **unlikely has ADHD** with {(1-proba):.1%} probability.")
        
       
        
        # Technical details in expander
        with st.expander("🔧Technical Details"):
            st.write(f"**Model:** Logistic Regression")
            st.write(f"**Features processed:** {input_df.shape[1]}")
            st.write(f"**Raw probability:** {proba:.4f}")
            st.write(f"**Decision threshold:** {threshold:.4f}")
            st.write(f"**Prediction:** {'ADHD' if prediction == 1 else 'No ADHD'}")
        
    except Exception as e:
        st.error(f" Error during prediction: {str(e)}")
        st.error("Please check that all inputs are valid and try again.")
        
        # Debug information in expander
        with st.expander(" Debug Information"):
            st.write(f"Input DataFrame shape: {input_df.shape}")
            st.write(f"Input DataFrame columns: {list(input_df.columns)}")
            st.write("Error details:", str(e))