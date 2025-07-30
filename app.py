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

# --- Backend Functions and Setup (Loaded only once) ---
# These functions handle the machine learning model and text processing.
# They are cached to ensure they only run once per session for better performance.

@st.cache_resource
def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    packages = ['punkt', 'stopwords', 'wordnet']
    for package in packages:
        try:
            # Check if the package is available
            if package == 'punkt':
                # The 'punkt' resource requires both directories. The 'punkt_tab'
                # directory was causing a LookupError in some cloud environments.
                # This updated check ensures both are present before skipping a download.
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('tokenizers/punkt_tab')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            # If not available, download it quietly
            nltk.download(package, quiet=True)

@st.cache_resource
def load_model_and_vectorizer():
    """
    Loads the pre-trained machine learning model and the TF-IDF vectorizer.
    Ensure 'phishing_detector_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.
    """
    try:
        with open('phishing_detector_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please ensure they are in the correct directory.")
        return None, None

def preprocess_text(text):
    """
    Cleans and prepares email text for the model by:
    1. Removing URLs, HTML tags, and non-alphabetic characters.
    2. Converting to lowercase.
    3. Tokenizing the text.
    4. Removing stopwords.
    5. Lemmatizing tokens.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove URLs, emails, and other clutter
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text) # remove email addresses
    text = re.sub(r'<.*?>', '', text) # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove punctuation and numbers
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

# --- Global Styles and Page Configuration ---
def apply_global_styles():
    """Applies modern, clean CSS styling to the entire application."""
    st.markdown(
        """
        <style>
            /* --- General App Styling --- */
            .stApp {
                background-color: #0F172A; /* Dark Slate Blue Background */
                color: #E2E8F0; /* Light Slate Gray Text */
            }

            /* --- Typography --- */
            h1, h2, h3 {
                font-family: 'Roboto', 'Inter', sans-serif;
                color: #FFFFFF; /* White headings */
            }

            /* --- Buttons --- */
            .stButton>button {
                font-size: 1.1rem;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                border: 1px solid #38BDF8; /* Sky Blue Border */
                background-color: transparent;
                color: #38BDF8; /* Sky Blue Text */
                transition: all 0.3s ease-in-out;
                font-weight: 600;
            }
            .stButton>button:hover {
                background-color: #38BDF8;
                color: #0F172A; /* Dark background on hover */
                transform: scale(1.02);
                box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
            }
            .stButton>button:active {
                transform: scale(0.98);
            }

            /* --- Info/Welcome Box --- */
            .welcome-box {
                background-color: #1E293B; /* Darker Slate */
                border-left: 5px solid #38BDF8;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 2rem;
            }

            /* --- Team Member Cards --- */
            .team-member-card {
                text-align: center;
                padding: 1.5rem 1rem;
                background-color: #1E293B;
                border-radius: 10px;
                border: 1px solid #334155; /* Slate Border */
                transition: all 0.3s ease;
            }
            .team-member-card:hover {
                transform: translateY(-5px);
                border-color: #38BDF8;
            }
            .team-member-card b {
                font-size: 1.2rem;
                color: #FFFFFF;
            }
            .team-member-card span {
                font-size: 0.9rem;
                color: #94A3B8; /* Lighter Slate for role */
            }

            /* --- Result Styling --- */
            .stAlert {
                border-radius: 8px;
                font-size: 1.2rem;
            }
            .stAlert [data-testid="stMarkdownContainer"] p {
                font-weight: 600;
            }

            /* --- Tabs Styling --- */
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                background-color: transparent;
                padding: 0 10px;
                border-radius: 8px;
            }
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #1E293B;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #38BDF8;
                color: #0F172A;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] p {
                color: #0F172A;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Page 1: Introduction Page ---
def intro_page():
    st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")
    apply_global_styles()

    st.title("🛡️ Welcome to PhishGuard")
    st.header("Your AI-Powered Phishing Detector")

    st.markdown(
        """
        <div class="welcome-box">
        We are a team of digital security enthusiasts dedicated to making the online world safer.
        This Phishing Detector is our flagship project, leveraging machine learning to shield you from malicious emails.
        Navigate with confidence.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Meet The Team")

    # --- Updated Team Members Section ---
    team_members = {
        "Hardik Jain": "Lead Developer",
        "Divyaraj Rajpurohit": "UI/UX & Frontend",
    }

    cols = st.columns(len(team_members))
    for i, (name, role) in enumerate(team_members.items()):
        with cols[i]:
            st.markdown(f"""
            <div class='team-member-card'>
                <b>{name}</b><br>
                <span>{role}</span>
            </div>
            """, unsafe_allow_html=True)

    st.write("") # Spacer
    st.write("") # Spacer

    if st.button("Launch PhishGuard Detector →", use_container_width=True):
        st.session_state.page = 'detector'
        st.rerun()

# --- Page 2: Phishing Detector ---
def detector_page():
    st.set_page_config(page_title="PhishGuard Detector", page_icon="🔎", layout="wide")
    apply_global_styles()

    if st.button("← Back to Welcome Page"):
        st.session_state.page = 'intro'
        st.rerun()

    st.title("🔎 PhishGuard AI Detector")
    st.write(
        "Analyze email content for potential phishing threats. "
        "Paste the email text or upload a file to begin."
    )

    model, vectorizer = load_model_and_vectorizer()
    if not model or not vectorizer:
        return # Stop execution if model loading failed

    def classify_text(input_text):
        """Processes and classifies the final extracted text, showing results."""
        with st.spinner('Analyzing text...'):
            processed_text = preprocess_text(input_text)
            if len(processed_text.split()) < 3:
                st.warning("Could not extract enough text for a reliable classification. Please provide more content.")
                return

            vectorized_text = vectorizer.transform([processed_text])
            # Create a feature for URL count
            url_count = len(re.findall(r'http\S+|www\S+|https\S+', input_text))
            urls_feature = scipy.sparse.csr_matrix([[url_count]])
            # Combine text features with URL count feature
            combined_features = scipy.sparse.hstack([vectorized_text, urls_feature])

            prediction = model.predict(combined_features)[0]
            confidence_scores = model.predict_proba(combined_features)[0]

            if prediction == 1:
                confidence = confidence_scores[1]
                st.error(f"**Result: PHISHING DETECTED** (Confidence: {confidence:.2%})", icon="🚨")
            else:
                confidence = confidence_scores[0]
                st.success(f"**Result: LIKELY LEGITIMATE** (Confidence: {confidence:.2%})", icon="✅")

    tab1, tab2 = st.tabs(["Paste Email Text", "Upload File (Text or Image)"])

    with tab1:
        email_text = st.text_area("Enter the full email text here:", height=300, placeholder="Dear customer, your account has been suspended. Please click here to verify...")
        if st.button("Analyze Pasted Text", use_container_width=True):
            if email_text:
                classify_text(email_text)
            else:
                st.warning("Please enter some text to analyze.")

    with tab2:
        uploaded_file = st.file_uploader("Upload a .txt, .png, .jpg, or .jpeg file", type=["txt", "png", "jpg", "jpeg"])
        if uploaded_file:
            try:
                if "text" in uploaded_file.type:
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    st.text_area("File Content:", file_content, height=300)
                    if st.button("Analyze Uploaded File", use_container_width=True):
                        classify_text(file_content)
                elif "image" in uploaded_file.type:
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded Image', use_container_width=True)
                    if st.button("Analyze Text from Image", use_container_width=True):
                        with st.spinner('Extracting text from image...'):
                            extracted_text = pytesseract.image_to_string(image)
                        if not extracted_text.strip():
                            st.warning("Could not detect any text in the uploaded image.")
                        else:
                            st.text_area("Extracted Text:", extracted_text, height=150)
                            classify_text(extracted_text)
            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract Error: Tesseract is not installed or not in your system's PATH. Please install it to use the image analysis feature.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

    st.markdown("---")
    st.write("Built with Streamlit & Scikit-learn | Designed by Hardik Jain & Divyaraj Rajpurohit")

# --- Main App Router ---
def main():
    """Main function to run the Streamlit app."""
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'intro'

    # Download NLTK data at the start
    download_nltk_data()

    # Display the current page based on session state
    if st.session_state.page == 'intro':
        intro_page()
    else:
        detector_page()

if __name__ == "__main__":
    main()
