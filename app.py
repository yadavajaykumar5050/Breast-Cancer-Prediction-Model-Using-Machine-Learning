import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align:center;'>Breast Cancer Prediction Model By Ajay</h1>",
    unsafe_allow_html=True
)

# Image (optional)
st.image("img.jpg", use_container_width=True)

st.markdown("---")

# Input
st.subheader("Input Breast Cancer Features")
st.write("Enter features separated by commas ( , )")

features_input = st.text_input("Features", placeholder="e.g. 17.99,10.38,122.8,...")

# Predict button
if st.button("Predict"):
    try:
        # Convert input to numpy array
        features = features_input.split(",")
        np_features = np.asarray(features, dtype=np.float32).reshape(1, -1)

        # Prediction
        pred = model.predict(np_features)

        # Result
        if pred[0] == 1:
            st.error("🚨 Cancerous")
            st.image("alert_imge.png", width=300)
            st.write(
                "⚠️ *Alert! You may have breast cancer. Please consult a doctor immediately.*"
            )
        else:
            st.success("✅ Not Cancerous")
            st.image("okay_img.jpg", width=300)
            st.write(
                "🎉 *Good news! You do not have breast cancer. Stay healthy and happy.*"
            )

    except Exception as e:
        st.warning("❌ Invalid input format. Please enter numeric values separated by commas.")
        st.text(str(e))