import streamlit as st
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))

cv = pickle.load(open('vectorizer.pkl', 'rb'))


st.title("Email Spam Classification App")
st.write("""
This application uses Machine Learning to classify emails as **Spam** or **Ham (Not Spam)**. 
You can enter an email, and it will predict if the email is spam or not based on its content.
""")

# User input
user_input = st.text_area("Enter the email content here:", height=150)

# Classify button
if st.button("Classify Email"):
    if user_input:  # Ensure input is provided
        # Pre-process input data using the loaded vectorizer
        vectorized_input = cv.transform([user_input]).toarray()

        # Make prediction
        prediction = model.predict(vectorized_input)

        # Display result based on prediction
        if prediction[0] == 0:
            st.write("🚫 **This email is NOT spam.** 📨")
        else:
            st.write("⚠️ **This email is spam!** 🚨")

    else:
        st.warning("Please enter an email to classify.")



st.sidebar.header("About This Application")
st.sidebar.write("""
Made by **Tarra Nikhitha**.
""")


# Disclaimer
st.sidebar.markdown("""
**Disclaimer**: This is a demonstration app. The model predictions are based on data it was trained on and may not always be perfect.
""")

