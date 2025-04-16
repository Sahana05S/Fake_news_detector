import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")
st.subheader("Check if a news article is Real or Fake")

# Text input
text = st.text_area("Enter news headline or article content:")

# Model selector
model_choice = st.radio("Choose model to use:", ["ISOT", "LIAR"])
model_key = "isot" if model_choice == "ISOT" else "liar"

# On submit
if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some news text.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json={"text": text, "model": model_key}
            )
            result = response.json()

            if result.get("result") == "Real":
                st.success(f"✅ Prediction: {result['result']}")
            elif result.get("result") == "Fake":
                st.error(f"❌ Prediction: {result['result']}")
            else:
                st.warning("⚠️ Could not classify the text.")

        except Exception as e:
            st.error(f"Error contacting the backend: {e}")
