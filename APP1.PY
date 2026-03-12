import streamlit as st
from PIL import Image
import pytesseract
import torch
import torch.nn.functional as F
import joblib
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difflib import SequenceMatcher

# ======================================
# UI
# ======================================

st.set_page_config(page_title="HackSphere Phishing Detector", layout="centered")

st.title("🛡 AI-Based Phishing & Scam Detection Model")
st.write("Upload a screenshot to analyze phishing risk.")

# ======================================
# Load Models
# ======================================

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained("models/spam_classifier")
    spam_model = AutoModelForSequenceClassification.from_pretrained("models/spam_classifier")

    phishing_model = joblib.load("models/phishing_model.pkl")

    return tokenizer, spam_model, phishing_model


tokenizer, spam_model, phishing_model = load_models()

# ======================================
# Legitimate Domains
# ======================================

legit_domains = [
    "paypal.com",
    "google.com",
    "amazon.com",
    "apple.com",
    "facebook.com",
    "microsoft.com"
]

# ======================================
# Filters to Reduce False Positives
# ======================================

def is_email(line):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", line))


def mostly_numbers(line):
    digits = sum(c.isdigit() for c in line)
    return digits / max(len(line),1) > 0.5


def too_short(line):
    return len(line.split()) < 4


phishing_keywords = [
    "verify","login","password","account","bank","urgent",
    "suspended","confirm","security","update","paypal"
]


def contains_keywords(line):
    return any(k in line.lower() for k in phishing_keywords)

# ======================================
# URL Similarity Detection
# ======================================

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def detect_fake_domain(domain):

    for legit in legit_domains:

        score = similarity(domain, legit)

        if score > 0.85 and domain != legit:
            return True, legit, score

        # strong phishing signal
        if legit.split(".")[0] in domain and domain != legit:
            return True, legit, score

    return False, None, None


# ======================================
# Extract URLs
# ======================================

def extract_urls(text):

    pattern = r"(https?://[^\s]+|www\.[^\s]+)"

    return re.findall(pattern, text)


# ======================================
# URL Feature Extraction
# ======================================

def extract_url_features(url):

    features = []

    features.append(len(url))
    features.append(url.count("."))
    features.append(url.count("-"))
    features.append(1 if "@" in url else 0)
    features.append(1 if "https" in url else 0)

    return np.array(features).reshape(1, -1)


# ======================================
# Upload Screenshot
# ======================================

uploaded_file = st.file_uploader(
    "Upload Screenshot",
    type=["png", "jpg", "jpeg"]
)

# ======================================
# Detection Pipeline
# ======================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Screenshot", width="stretch")

    st.subheader("🔎 OCR Text Extraction")

    extracted_text = pytesseract.image_to_string(image)

    st.text_area("Extracted Text", extracted_text, height=200)

    reasons = []
    max_prob = 0

    suspicious_domains_found = False
    url_model_flag = False

    # ======================================
    # NLP Line Analysis
    # ======================================

    lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]

    for i, line in enumerate(lines):

        if is_email(line) or mostly_numbers(line) or too_short(line):
            continue

        if not contains_keywords(line):
            continue

        inputs = tokenizer(
            line,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():

            outputs = spam_model(**inputs)

            logits = outputs.logits

            probs = F.softmax(logits, dim=1)

            phishing_prob = probs[0][1].item()

        max_prob = max(max_prob, phishing_prob)

        if phishing_prob > 0.50:

            reasons.append(
                f'Line {i+1}: "{line}" → suspicious phishing language ({round(phishing_prob*100,2)}%)'
            )

    # ======================================
    # URL Detection
    # ======================================

    urls = extract_urls(extracted_text)

    for url in urls:

        domain = url.replace("https://", "").replace("http://", "").split("/")[0]

        suspicious, legit, score = detect_fake_domain(domain)

        if suspicious:

            suspicious_domains_found = True

            reasons.append(
                f"Suspicious domain **{domain}** resembles **{legit}**"
            )

        try:

            features = extract_url_features(url)

            pred = phishing_model.predict(features)[0]

            if pred == 1:

                url_model_flag = True

                reasons.append(
                    f"URL ML model flagged **{url}** as phishing"
                )

        except:
            pass

    # ======================================
    # Risk Score System
    # ======================================

    risk_score = 0

    if max_prob > 0.50:
        risk_score += 1

    if suspicious_domains_found:
        risk_score += 2

    if url_model_flag:
        risk_score += 2

    # ======================================
    # Risk Level
    # ======================================

    if risk_score >= 3:
        risk = "HIGH"
    elif risk_score >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ======================================
    # Final Result
    # ======================================

    st.subheader("🧠 Detection Result")

    st.write("Risk Level:", risk)

    st.write("Highest Phishing Probability:", round(max_prob*100,2), "%")

    if risk_score >= 2:

        st.error("⚠ PHISHING DETECTED")

        st.subheader("🚨 Detection Reasons")

        for r in reasons:
            st.write("•", r)

    else:

        st.success("✅ SAFE CONTENT")

        st.write("No phishing indicators detected.")

    # ======================================
    # Technical Details
    # ======================================

    st.subheader("🔬 Analysis Details")

    st.write("Detected URLs:", urls)