import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Healthcare - Multiple Disease Prediction", layout="wide")
st.title("🏥 Healthcare - Multiple Disease Prediction")

# ===============================
# SIDEBAR OPTIONS
# ===============================
st.sidebar.header("Select a Disease")

disease_options = {
    "Parkinson's": {
        "file": "Park_knn.pkl",  # saved as tuple (scaler, knn)
        "icon": "/Users/gokulravindran/Desktop/Guvi-Projects/MultplDS-ML/Rawfile/park_icon.webp",
        "features": [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
            'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ],
        "healthy": [150, 200, 100, 0.002, 0.00001, 0.001, 0.001, 0.003, 0.02, 0.2, 0.01, 0.02, 0.02, 0.03, 0.01, 20, 0.4, 0.8, -5, 0.2, 2, 0.1],
        "suggestion": "Maintain a balanced diet rich in antioxidants, do regular physiotherapy, practice voice exercises, and ensure good sleep.",
        "scale": True,
        "no_disease_class": 0
    },
    "Kidney": {
        "file": "kid_decisiontree.pkl",
        "icon": "/Users/gokulravindran/Desktop/Guvi-Projects/MultplDS-ML/Rawfile/kid_icon.webp",
        "features": [
            'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
            'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ],
        "healthy": [80, 1.02, 0, 0, 0, 0, 0, 0, 110, 15, 1.2, 140, 4.5, 15, 45, 8000, 5, 0, 0, 0, 1, 0, 0],
        "suggestion": "Stay hydrated, reduce salt intake, avoid processed foods, manage blood pressure and sugar levels, and include kidney-friendly foods.",
        "scale": False,
        "no_disease_class": 0
    },
    "Liver": {
        "file": "lvr_randomforest.pkl",
        "icon": "/Users/gokulravindran/Desktop/Guvi-Projects/MultplDS-ML/Rawfile/lvr_icon.webp",
        "features": ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos', 'Sgpt', 'Sgot', 'Total_Protiens', 'Albumin', 'AG_ratio'],
        "healthy": [0.8, 0.2, 120, 25, 30, 7, 4.5, 1.2],
        "suggestion": "Avoid alcohol, maintain a healthy weight, eat high-fiber foods, exercise regularly, and include antioxidant-rich foods.",
        "scale": False,
        "no_disease_class": 2
    }
}

choice = st.sidebar.radio("Choose a Disease Model:", disease_options.keys())

# ===============================
# LOAD MODEL
# ===============================
model_info = disease_options[choice]

with open(model_info["file"], "rb") as f:
    loaded = pickle.load(f)

# Handle KNN with scaler tuple or other models
if model_info.get("scale"):
    # Expecting tuple (scaler, model)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        scaler, model = loaded
    else:
        st.error("KNN model must be saved as (scaler, model) tuple!")
        st.stop()
else:
    model = loaded
    scaler = None

# ===============================
# DISPLAY DISEASE ICON
# ===============================
st.subheader(f"🔬 Predicting for: {choice}")
st.image(model_info["icon"], width=150)

# ===============================
# INPUT FIELDS
# ===============================
inputs = []
st.markdown("### Please enter the following details:")
cols = st.columns(2)
for i, feat in enumerate(model_info["features"]):
    with cols[i % 2]:
        val = st.number_input(f"{feat}", value=float(model_info["healthy"][i]))
        inputs.append(val)

inputs_array = np.array(inputs).reshape(1, -1)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict"):
    # Apply scaling if required
    if scaler is not None:
        inputs_array_scaled = scaler.transform(inputs_array)
    else:
        inputs_array_scaled = inputs_array

    prediction = model.predict(inputs_array_scaled)[0]
    prob = model.predict_proba(inputs_array_scaled)[0]

    # Determine risk level based on predicted class
    if prediction == model_info.get("no_disease_class", 0):
        risk_level = "Low"
        color = "green"
    else:
        max_prob = max(prob)
        if max_prob < 0.4:
            risk_level = "Low"
            color = "green"
        elif max_prob < 0.7:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"

    # Display results
    st.markdown("## 🩺 Prediction Result")
    st.write(f"**Predicted Outcome:** {prediction}")
    st.write(f"**Likelihood (Probabilities):** {prob}")
    st.write(f"**Health Risk Level:** {risk_level}")

    # ===============================
    # VISUALIZATION: Human Body Outline
    # ===============================
    st.markdown("### Health Risk Visualization")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
        <svg width="200" height="400">
        <rect x="50" y="20" width="100" height="300" rx="50" ry="50" style="fill:{color};stroke:black;stroke-width:2;" />
        </svg>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===============================
    # FEATURE COMPARISON BAR CHART
    # ===============================
    st.markdown("### 📊 Feature Comparison")
    feature_df = pd.DataFrame({
        "Feature": model_info["features"],
        "Healthy": model_info["healthy"],
        "You": inputs
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    feature_df.set_index("Feature").plot(kind="bar", ax=ax)
    plt.xticks(rotation=90)
    plt.ylabel("Value")
    plt.title("Comparison with Healthy Reference")
    st.pyplot(fig)

    # ===============================
    # SUGGESTIONS
    # ===============================
    st.markdown("### 💡 Suggestion:")
    st.write(model_info["suggestion"])






