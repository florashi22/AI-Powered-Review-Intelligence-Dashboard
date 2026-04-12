import streamlit as st

st.title("AI-Powered Review Intelligence Dashboard")

st.write("Predict Yelp-style ratings from customer reviews.")

review = st.text_area("Enter a review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        st.success("Model will predict here (coming soon)")
