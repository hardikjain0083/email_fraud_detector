import streamlit as st
import pickle
import re
import os
import scipy.sparse
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from PIL import Image
import pytesseract
import base64

# --- Tesseract Configuration ---
# Assumes Tesseract is installed and in the system's PATH.
# If not, add the path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Asset Function ---
# Function to encode local images for CSS background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Backend Functions and Setup (Loaded only once) ---

@st.cache_resource
def download_nltk_data():
    """Downloads required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

@st.cache_resource
def load_model_and_vectorizer():
    """Loads the pre-trained model and vectorizer."""
    model_path = 'phishing_detector_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)
    return loaded_model, loaded_vectorizer

def preprocess_text(text):
    """Cleans and prepares email text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

# --- Page 1: Introduction Page (Redesigned) ---

def intro_page():
    st.set_page_config(page_title="Team Pyrates", page_icon="🏴‍☠️", layout="centered")
    
    # Custom CSS for the pirate theme
    st.markdown(
        """
        <style>
            .stApp {
                background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAQUlEQVQYV2NkYGD4z0A6+JzBysL4g2E4fXiG4TRhBGb8/v//DAwM/wH5//8/w38GIAXDf4b/DAwM/wH5//8/w38GIAXDf4b/DAwM/wH5//8/w38GIAXDDwYAINcMAAAAAElFTkSuQmCC);
                background-color: #0a0a1a;
                color: #f0f0f0;
            }
            .stTitle, .stHeader {
                font-family: 'Courier New', Courier, monospace;
                color: #ffc107;
                text-shadow: 2px 2px 4px #000000;
            }
            .stButton>button {
                font-size: 1.2rem;
                padding: 0.8rem 2rem;
                border-radius: 12px;
                border: 2px solid #ffc107;
                background-color: transparent;
                color: #ffc107;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #ffc107;
                color: #0a0a1a;
            }
            .stInfo {
                background-color: rgba(0, 0, 0, 0.5);
                border-left: 5px solid #ffc107;
                padding: 1rem;
                border-radius: 8px;
            }
            .team-member {
                text-align: center;
                padding: 1rem;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                border: 1px solid #ffc107;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Welcome Aboard, Matey! 🏴‍☠️")
    st.header("A Project by Team Pyrates")
    
    st.info(
        "We are **Team Pyrates**, a crew of digital adventurers navigating the treacherous seas of cyberspace. "
        "Our mission is to build tools that protect users from online threats. This Phishing Detector is our flagship project, "
        "designed to help you identify and avoid malicious emails."
    )
    
    st.write("---")
    st.subheader("Meet the Crew")

    # --- UPDATE THIS SECTION WITH YOUR TEAM MEMBERS ---
    team_members = {
        "Hardik": "Captain & Lead Developer ⚓",
        "Member 2": "First Mate & UI/UX Designer 🎨",
        "Member 3": "Navigator & Data Scientist 🗺️",
        "Member 4": "Quartermaster & Backend Engineer ⚙️"
    }
    
    cols = st.columns(len(team_members))
    for i, (name, role) in enumerate(team_members.items()):
        with cols[i]:
            st.markdown(f"<div class='team-member'><b>{name}</b><br>{role}</div>", unsafe_allow_html=True)

    st.write("") # Spacer
    
    if st.button("Proceed to the Phishing Detector →", type="primary", use_container_width=True):
        st.session_state.page = 'detector'
        st.rerun()

# --- Page 2: Phishing Detector ---

def detector_page():
    st.set_page_config(page_title="Phishing Detector", page_icon="🎣", layout="wide")

    # --- NEW: Back button to return to the intro page ---
    if st.button("← Back to Intro Page"):
        st.session_state.page = 'intro'
        st.rerun()
    
    st.title("🎣 Phishing Email Detector")
    st.write(
        "This tool uses a machine learning model to analyze email content and predict whether it's a phishing attempt. "
        "Paste the full text of an email or upload a file to check."
    )

    model, vectorizer = load_model_and_vectorizer()

    def classify_text(input_text):
        """Processes and classifies the final extracted text."""
        processed_text = preprocess_text(input_text)
        if len(processed_text.split()) < 3:
            st.warning("Could not extract enough text for a reliable classification.")
        else:
            vectorized_text = vectorizer.transform([processed_text])
            url_count = len(re.findall(r'http\S+|www\S+|https\S+', input_text))
            urls_feature = scipy.sparse.csr_matrix([[url_count]])
            combined_features = scipy.sparse.hstack([vectorized_text, urls_feature])
            prediction = model.predict(combined_features)[0]
            confidence_scores = model.predict_proba(combined_features)[0]
            if prediction == 1:
                st.error(f"**Result: PHISHING** (Confidence: {confidence_scores[1]:.2%})")
            else:
                st.success(f"**Result: LEGITIMATE** (Confidence: {confidence_scores[0]:.2%})")

    tab1, tab2 = st.tabs(["Paste Email Text", "Upload File (Text or Image)"])

    with tab1:
        email_text = st.text_area("Enter email text here:", height=250, placeholder="Dear customer, please verify your account...")
        if st.button("Classify Pasted Text", type="primary", use_container_width=True):
            if email_text:
                classify_text(email_text)
            else:
                st.warning("Please enter some text to classify.")

    with tab2:
        uploaded_file = st.file_uploader("Upload a .txt, .png, .jpg, or .jpeg file", type=["txt", "png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                if "text" in uploaded_file.type:
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    st.text_area("File Content:", file_content, height=250)
                    if st.button("Classify Uploaded File", type="primary", use_container_width=True):
                        classify_text(file_content)
                elif "image" in uploaded_file.type:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image.', use_container_width=True)
                    if st.button("Classify Text from Image", type="primary", use_container_width=True):
                        extracted_text = pytesseract.image_to_string(image)
                        if not extracted_text.strip():
                            st.warning("Could not detect any text in the uploaded image.")
                        else:
                            st.text_area("Extracted Text:", extracted_text, height=150)
                            classify_text(extracted_text)
            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract Error: Tesseract is not installed or not in your system's PATH.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    st.markdown("---")
    st.write("Built by Team Pyrates with Streamlit and Scikit-learn.")


# --- Main App Router ---

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

# Download NLTK data at the start
download_nltk_data()

# Display the current page
if st.session_state.page == 'intro':
    intro_page()
else:
    detector_page()
