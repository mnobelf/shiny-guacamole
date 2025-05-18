import streamlit as st
import pandas as pd
import joblib

# Load CSV
raw_data = pd.read_csv(
    "data.csv",
    sep=';'
)

# Define target and feature types (same as notebook)
target_col = 'Target'
categorical_features = [
    'Marital status','Application mode','Course','Daytime/evening attendance\t',
    'Previous qualification','Nacionality','Mother\'s qualification','Father\'s qualification',
    'Mother\'s occupation','Father\'s occupation','Displaced','Educational special needs',
    'Debtor','Tuition fees up to date','Gender','Scholarship holder','International'
]
numeric_features = [col for col in raw_data.columns if col not in categorical_features + [target_col]]

# Load artifacts
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
target_map = joblib.load('target_map.pkl')
inv_target = {v:k for k,v in target_map.items()}

st.title('Student Status Prediction')
st.write("Input features to predict: Dropout, Enrolled, or Graduate.")

# Build inputs
df_defaults = raw_data.drop(columns=[target_col])
input_dict = {}
for col in df_defaults.columns:
    if col in numeric_features:
        input_dict[col] = st.number_input(col, value=float(df_defaults[col].mean()))
    else:
        opts = sorted(df_defaults[col].unique())
        input_dict[col] = st.selectbox(col, opts)

input_df = pd.DataFrame([input_dict])
X_prep = preprocessor.transform(input_df)
pred_idx = model.predict(X_prep)[0]
probs = model.predict_proba(X_prep)[0]

st.subheader('Prediction')
st.write(f"**{inv_target[pred_idx]}**")
st.write("Class Probabilities:")
st.table(pd.DataFrame({'Class': list(inv_target.values()), 'Probability': probs}))