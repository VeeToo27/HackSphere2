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

# ======================================
# CSS Styling
# ======================================

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-dark:      #0a0c10;
        --bg-card:      #10141c;
        --bg-panel:     #141820;
        --accent-cyan:  #00e5ff;
        --accent-red:   #ff3c5a;
        --accent-green: #00e676;
        --accent-amber: #ffab00;
        --text-primary: #e8ecf4;
        --text-muted:   #8892a4;
        --border:       #1e2535;
        --border-glow:  rgba(0, 229, 255, 0.25);
        --font-display: 'Rajdhani', sans-serif;
        --font-mono:    'IBM Plex Mono', monospace;
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-dark);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,229,255,0.07) 0%, transparent 70%),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 39px,
                rgba(255,255,255,0.015) 39px,
                rgba(255,255,255,0.015) 40px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 39px,
                rgba(255,255,255,0.015) 39px,
                rgba(255,255,255,0.015) 40px
            );
        font-family: var(--font-mono);
        color: var(--text-primary);
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Main container ── */
    .block-container {
        max-width: 780px;
        padding: 3rem 2rem 5rem;
    }

    /* ── Page Title ── */
    h1 {
        font-family: var(--font-display) !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        color: var(--accent-cyan) !important;
        text-shadow: 0 0 28px rgba(0,229,255,0.45);
        margin-bottom: 0.2rem !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        font-family: var(--font-display) !important;
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        color: var(--text-primary) !important;
        border-left: 3px solid var(--accent-cyan);
        padding-left: 0.75rem !important;
        margin-top: 2rem !important;
        text-transform: uppercase;
    }

    /* ── Paragraph / body text ── */
    p, .stMarkdown p {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.7;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 1px dashed var(--accent-cyan);
        border-radius: 6px;
        padding: 1.5rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 0 18px var(--border-glow);
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
    }

    /* ── Image caption ── */
    [data-testid="stImage"] figcaption {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-family: var(--font-mono) !important;
        text-align: center;
    }

    /* ── Text area (OCR output) ── */
    textarea {
        background: var(--bg-panel) !important;
        color: #a8d8a8 !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.8rem !important;
        caret-color: var(--accent-cyan);
        box-shadow: inset 0 1px 8px rgba(0,0,0,0.4);
        transition: border-color 0.2s;
    }
    textarea:focus {
        border-color: var(--accent-cyan) !important;
        outline: none !important;
        box-shadow: 0 0 12px var(--border-glow), inset 0 1px 8px rgba(0,0,0,0.4) !important;
    }
    label[data-testid="stTextAreaLabel"] {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em;
    }

    /* ── Write outputs (Risk Level, URLs, etc.) ── */
    [data-testid="stText"],
    .stMarkdown {
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.88rem !important;
    }

    /* ── Success alert ── */
    [data-testid="stAlert"][kind="success"],
    div[class*="stSuccess"] {
        background: rgba(0, 230, 118, 0.08) !important;
        border: 1px solid var(--accent-green) !important;
        border-radius: 6px !important;
        color: var(--accent-green) !important;
        font-family: var(--font-display) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        box-shadow: 0 0 18px rgba(0,230,118,0.15);
    }

    /* ── Error alert ── */
    [data-testid="stAlert"][kind="error"],
    div[class*="stError"] {
        background: rgba(255, 60, 90, 0.08) !important;
        border: 1px solid var(--accent-red) !important;
        border-radius: 6px !important;
        color: var(--accent-red) !important;
        font-family: var(--font-display) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        box-shadow: 0 0 18px rgba(255,60,90,0.15);
    }

    /* ── Warning alert ── */
    [data-testid="stAlert"][kind="warning"],
    div[class*="stWarning"] {
        background: rgba(255, 171, 0, 0.08) !important;
        border: 1px solid var(--accent-amber) !important;
        border-radius: 6px !important;
        color: var(--accent-amber) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.88rem !important;
        box-shadow: 0 0 14px rgba(255,171,0,0.12);
    }

    /* ── Reason bullet items ── */
    .stMarkdown li,
    .stMarkdown ul {
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.85rem !important;
        line-height: 1.8 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb {
        background: #1e2b3a;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] {
        color: var(--accent-cyan) !important;
    }
</style>
""", unsafe_allow_html=True)

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
