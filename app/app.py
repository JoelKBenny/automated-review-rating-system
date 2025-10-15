# Loading Libraries
import streamlit as st
import joblib
import re
import string 
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
model_a = joblib.load(r"C:/Users/joelk/automated-review-rating-system/models/Model_A.pkl")
model_b = joblib.load(r"C:/Users/joelk/automated-review-rating-system/models/Model_B.pkl")

# Load vectorizers
vec_a = joblib.load(r"C:/Users/joelk/automated-review-rating-system/models/Vec_A.pkl")
vec_b = joblib.load(r"C:/Users/joelk/automated-review-rating-system/models/Vec_B.pkl")


# Preprocess function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_review(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode("ascii", "ignore").decode("ascii")
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# Streamlit UI
st.title("Automated Review Rating System")
st.write("Enter a product review and get the predicted rating from both models:")

user_review = st.text_area("Enter your review")

# Adding Predict button
if st.button("Predict"):
    if user_review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        cleaned_review = preprocess_review(user_review)

        # Transforming the cleaned review using preloaded vectorizers
        x_a = vec_a.transform([cleaned_review])
        x_b = vec_b.transform([cleaned_review])

        # MakING predictions
        pred_a = model_a.predict(x_a)[0]
        pred_b = model_b.predict(x_b)[0]

        # DisplayING results
        st.success(f"Predicted Rating by Model A (Balanced): {pred_a}⭐")
        st.success(f"Predicted Rating by Model B (Imbalanced): {pred_b}⭐")
