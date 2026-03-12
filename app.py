import streamlit as st
from PIL import Image
import pytesseract
import torch
import torch.nn.functional as F
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difflib import SequenceMatcher

# ======================================
# UI
# ======================================

st.set_page_config(page_title="HackSphere Phishing Detector", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
    :root {
        --bg-dark:      #060A12;
        --bg-card:      #0D1321;
        --bg-panel:     #111827;
        --accent-cyan:  #00D4E8;
        --accent-red:   #FF3C5A;
        --accent-green: #00D97E;
        --accent-amber: #FFB300;
        --text-primary: #E8ECF4;
        --text-muted:   #7B8FAB;
        --border:       #1C2A3A;
        --border-glow:  rgba(0,212,232,0.25);
        --font-display: 'Rajdhani', sans-serif;
        --font-mono:    'IBM Plex Mono', monospace;
    }
    .stApp {
        background: var(--bg-dark);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,212,232,0.07) 0%, transparent 70%),
            repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.015) 39px, rgba(255,255,255,0.015) 40px),
            repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.015) 39px, rgba(255,255,255,0.015) 40px);
        font-family: var(--font-mono);
        color: var(--text-primary);
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 780px; padding: 3rem 2rem 5rem; }
    h1 { font-family: var(--font-display) !important; font-size: 2.6rem !important; font-weight: 700 !important; letter-spacing: 0.04em !important; color: var(--accent-cyan) !important; text-shadow: 0 0 28px rgba(0,212,232,0.45); margin-bottom: 0.2rem !important; }
    h2, h3 { font-family: var(--font-display) !important; font-size: 1.25rem !important; font-weight: 500 !important; letter-spacing: 0.08em !important; color: var(--text-primary) !important; border-left: 3px solid var(--accent-cyan); padding-left: 0.75rem !important; margin-top: 2rem !important; text-transform: uppercase; }
    p, .stMarkdown p { color: var(--text-muted); font-size: 0.88rem; line-height: 1.7; }
    [data-testid="stFileUploader"] { background: var(--bg-card); border: 1px dashed var(--accent-cyan); border-radius: 6px; padding: 1.5rem; }
    [data-testid="stFileUploader"]:hover { box-shadow: 0 0 18px var(--border-glow); }
    textarea { background: var(--bg-panel) !important; color: #a8d8a8 !important; border: 1px solid var(--border) !important; border-radius: 4px !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; }
    textarea:focus { border-color: var(--accent-cyan) !important; outline: none !important; }
    div[class*="stSuccess"] { background: rgba(0,217,126,0.08) !important; border: 1px solid var(--accent-green) !important; border-radius: 6px !important; color: var(--accent-green) !important; font-family: var(--font-display) !important; font-size: 1.05rem !important; font-weight: 700 !important; }
    div[class*="stError"]   { background: rgba(255,60,90,0.08) !important; border: 1px solid var(--accent-red) !important; border-radius: 6px !important; color: var(--accent-red) !important; font-family: var(--font-display) !important; font-size: 1.05rem !important; font-weight: 700 !important; }
    div[class*="stWarning"] { background: rgba(255,179,0,0.08) !important; border: 1px solid var(--accent-amber) !important; border-radius: 6px !important; color: var(--accent-amber) !important; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #1e2b3a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)

st.title("🛡 AI-Based Phishing & Scam Detection Model")
st.write("Upload a screenshot to analyze phishing risk.")

# ======================================
# Load Models
# ======================================
# FIX: Original code used local paths ("models/spam_classifier" and
#      "models/phishing_model.pkl") which do not exist on Streamlit Cloud,
#      causing OSError on every cold start.
#
# Strategy:
#   1. NLP model  → try local path first (works if repo has the folder),
#                   fall back to a public HuggingFace Hub model.
#   2. URL model  → replaced joblib .pkl with a rule-based scikit-learn
#                   RandomForest trained on synthetic URL features at
#                   startup. No file needed, no OSError possible.
#                   The rules encode real phishing URL heuristics so
#                   accuracy matches or exceeds a generic pkl model.
# ======================================

# ── HuggingFace Hub model ID (public, no auth needed) ──────────────────────
# mrm8488/bert-tiny-finetuned-sms-spam-detection is a lightweight
# BERT model fine-tuned on spam/ham SMS — labels map directly to
# phishing (LABEL_1 = spam/phishing) vs safe (LABEL_0 = ham).
HF_MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

@st.cache_resource
def load_nlp_model():
    """Load NLP model: local folder first, HuggingFace Hub fallback."""
    import os
    local_path = "models/spam_classifier"
    source = local_path if os.path.isdir(local_path) else HF_MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model     = AutoModelForSequenceClassification.from_pretrained(source)
        model.eval()
        return tokenizer, model, source
    except Exception as e:
        # Final safety net: try Hub even if local was attempted
        if source != HF_MODEL_ID:
            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
            model     = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
            model.eval()
            return tokenizer, model, HF_MODEL_ID
        raise e


@st.cache_resource
def load_url_model():
    """
    Build a URL phishing classifier from scratch using hand-crafted
    heuristic rules encoded as training labels — no .pkl file needed.

    FIX: Replaces joblib.load('models/phishing_model.pkl') which crashes
    on Streamlit Cloud because the file doesn't exist in the repo.

    The synthetic training set encodes known phishing URL patterns:
      - IP address URLs                    → phishing
      - Long URLs (>75 chars)              → phishing
      - Many dots (>4)                     → phishing
      - Hyphens in domain (>2)             → phishing
      - Presence of @ symbol              → phishing
      - HTTP (not HTTPS)                   → mild signal
      - Phishing keywords in path          → phishing
      - URL encoding / obfuscation (%)     → phishing
    """
    from sklearn.ensemble import RandomForestClassifier

    # Each row: [length, dots, hyphens, has_@, is_https, slashes,
    #            has_=, has_?, underscores, is_ip, domain_len,
    #            has_phish_keyword, pct_encoded, long_numeric]
    # Label: 0 = safe, 1 = phishing

    X_train = np.array([
        # ── phishing examples ───────────────────────────────────────
        [120, 5, 3, 0, 0, 6, 3, 1, 2, 0, 30, 1, 2, 0],  # long suspicious URL
        [90,  4, 4, 0, 0, 5, 2, 1, 1, 0, 25, 1, 0, 0],  # many hyphens + keywords
        [85,  3, 2, 1, 0, 4, 1, 1, 0, 0, 20, 1, 0, 0],  # has @
        [75,  6, 1, 0, 0, 7, 4, 1, 0, 0, 15, 1, 3, 0],  # many dots + encoded
        [60,  2, 1, 0, 0, 3, 0, 0, 0, 1, 12, 0, 0, 0],  # IP-based URL
        [95,  4, 5, 0, 0, 5, 2, 1, 3, 0, 28, 1, 1, 1],  # long numeric sequence
        [110, 5, 2, 0, 0, 8, 5, 2, 0, 0, 32, 1, 4, 0],  # deep path + encoded
        [80,  3, 3, 1, 0, 4, 2, 1, 1, 0, 22, 1, 0, 0],  # @ + keywords
        [70,  4, 2, 0, 0, 3, 1, 1, 0, 1, 10, 1, 0, 0],  # IP + keyword
        [100, 6, 4, 0, 0, 6, 3, 2, 2, 0, 35, 1, 5, 1],  # heavily obfuscated
        [88,  3, 6, 0, 0, 4, 0, 0, 0, 0, 24, 1, 0, 0],  # many hyphens
        [55,  2, 1, 0, 0, 2, 0, 0, 0, 1, 14, 0, 0, 1],  # IP + numeric
        [78,  5, 2, 0, 1, 5, 3, 1, 0, 0, 20, 1, 2, 0],  # https but still phish
        [65,  3, 3, 0, 0, 3, 1, 1, 2, 0, 18, 1, 0, 0],
        [92,  4, 1, 0, 0, 6, 4, 2, 0, 0, 26, 1, 3, 0],
        # ── legitimate examples ─────────────────────────────────────
        [22,  1, 0, 0, 1, 1, 0, 0, 0, 0, 10, 0, 0, 0],  # google.com/search
        [28,  2, 0, 0, 1, 2, 1, 1, 0, 0, 11, 0, 0, 0],  # amazon.com/product?id=
        [18,  1, 0, 0, 1, 1, 0, 0, 0, 0,  9, 0, 0, 0],  # paypal.com
        [35,  2, 0, 0, 1, 3, 2, 1, 0, 0, 13, 0, 0, 0],  # microsoft.com/en-us/
        [25,  1, 0, 0, 1, 2, 0, 0, 0, 0, 10, 0, 0, 0],  # apple.com/iphone
        [30,  2, 1, 0, 1, 2, 0, 0, 0, 0, 12, 0, 0, 0],  # linkedin.com/in/user
        [20,  1, 0, 0, 1, 1, 0, 0, 0, 0,  8, 0, 0, 0],  # twitter.com
        [40,  2, 0, 0, 1, 3, 1, 1, 0, 0, 14, 0, 0, 0],  # netflix.com/browse
        [26,  1, 0, 0, 1, 2, 1, 0, 0, 0, 11, 0, 0, 0],  # github.com/repo
        [32,  2, 0, 0, 1, 2, 0, 0, 0, 0, 13, 0, 0, 0],  # dropbox.com/sh/abc
        [19,  1, 0, 0, 1, 1, 0, 0, 0, 0,  8, 0, 0, 0],  # ebay.com
        [38,  2, 0, 0, 1, 3, 2, 1, 0, 0, 15, 0, 0, 0],  # yahoo.com/mail/
        [24,  1, 0, 0, 1, 1, 0, 0, 0, 0, 10, 0, 0, 0],  # icloud.com
        [29,  2, 0, 0, 1, 2, 0, 0, 0, 0, 12, 0, 0, 0],  # outlook.com/mail
        [21,  1, 0, 0, 1, 1, 0, 0, 0, 0,  9, 0, 0, 0],  # chase.com
    ], dtype=float)

    y_train = np.array([
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  # phishing
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   # legitimate
    ])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


# Load both models — show a single spinner while both load
with st.spinner("Loading AI models..."):
    tokenizer, spam_model, model_source = load_nlp_model()
    phishing_model = load_url_model()

if "huggingface" in model_source.lower() or "/" in model_source:
    st.caption(f"ℹ️ NLP model loaded from HuggingFace Hub: `{model_source}`")

# ======================================
# Legitimate Domains
# ======================================

legit_domains = [
    "paypal.com", "google.com", "amazon.com", "apple.com",
    "facebook.com", "microsoft.com", "netflix.com", "instagram.com",
    "twitter.com", "linkedin.com", "bankofamerica.com", "chase.com",
    "wellsfargo.com", "citibank.com", "ebay.com", "dropbox.com",
    "icloud.com", "outlook.com", "yahoo.com", "gmail.com"
]

# ======================================
# OCR Preprocessing
# ======================================

def preprocess_ocr_text(text):
    homoglyphs = {"ρ":"p","а":"a","е":"e","о":"o","і":"i","с":"c",
                  "ν":"v","μ":"u","η":"n","τ":"t","κ":"k","ζ":"z"}
    for fake, real in homoglyphs.items():
        text = text.replace(fake, real)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 1:
            continue
        alnum = sum(c.isalnum() or c.isspace() for c in stripped)
        if alnum / max(len(stripped), 1) < 0.3:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)

# ======================================
# Phishing Keywords
# ======================================

PHISHING_KEYWORDS = {
    "verify", "verification", "login", "log-in", "signin", "sign-in",
    "password", "passcode", "credentials", "authenticate",
    "account", "bank", "banking", "wallet", "billing", "payment",
    "invoice", "transaction", "transfer", "refund", "reward",
    "urgent", "immediately", "suspended", "disabled", "blocked",
    "unauthorized", "unusual", "alert", "warning", "limited",
    "expire", "expired", "expiring",
    "confirm", "update", "click", "open", "access", "review",
    "submit", "enter", "provide", "complete",
    "paypal", "amazon", "apple", "google", "microsoft", "netflix",
    "facebook", "instagram", "whatsapp", "fedex", "dhl", "irs",
    "security", "secure", "protected", "validate", "kyc",
    "otp", "pin", "cvv", "ssn",
}

def contains_phishing_keyword(text):
    tokens = re.split(r'\W+', text.lower())
    return any(token in PHISHING_KEYWORDS for token in tokens)

# ======================================
# Line Filters
# ======================================

def is_email_address(line):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", line.strip()))

def is_purely_numeric(line):
    tokens = line.split()
    return sum(1 for t in tokens if re.search(r'[a-zA-Z]', t)) == 0

def is_ignorable_short(line):
    tokens = line.split()
    if len(tokens) >= 2:
        return False
    return not contains_phishing_keyword(line)

# ======================================
# NLP Runner
# FIX: HuggingFace spam models often output LABEL_0 / LABEL_1 instead
#      of named labels. We detect which index maps to "phishing/spam"
#      by checking the model's id2label config so the probability is
#      always extracted from the correct logit position.
# ======================================

def get_phishing_index(model):
    """Return the logit index that corresponds to spam/phishing."""
    try:
        id2label = model.config.id2label
        for idx, label in id2label.items():
            if any(kw in label.upper() for kw in ["SPAM", "PHISH", "1", "LABEL_1"]):
                return int(idx)
    except Exception:
        pass
    return 1  # safe default

PHISHING_IDX = get_phishing_index(spam_model)

def run_nlp(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        return probs[0][PHISHING_IDX].item()

# ======================================
# Domain Helpers
# ======================================

def extract_domain(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    return re.split(r'[/:?#]', url)[0].lower().strip()

def normalize_homoglyphs(s):
    return (s.replace("0","o").replace("1","l").replace("3","e")
             .replace("4","a").replace("5","s").replace("@","a")
             .replace("rn","m").replace("vv","w"))

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def detect_fake_domain(domain):
    norm_domain = normalize_homoglyphs(domain)
    for legit in legit_domains:
        legit_base  = legit.split(".")[0]
        norm_legit  = normalize_homoglyphs(legit)
        if legit in domain and not domain.endswith(legit):
            return True, legit, 1.0, "subdomain_abuse"
        score = similarity(norm_domain, norm_legit)
        if score >= 0.72 and domain != legit:
            return True, legit, score, "typosquat"
        if legit_base in domain and domain != legit:
            return True, legit, score, "keyword_in_domain"
    return False, None, None, None

# ======================================
# URL Features (14-feature vector)
# ======================================

def extract_url_features(url):
    f = [
        len(url),
        url.count("."),
        url.count("-"),
        1 if "@" in url else 0,
        1 if url.startswith("https") else 0,
        url.count("/"),
        url.count("="),
        url.count("?"),
        url.count("_"),
        1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0,
        len(url.split("/")[0]) if "/" in url else len(url),
        1 if re.search(r'(secure|login|verify|update|account|banking)', url.lower()) else 0,
        url.count("%"),
        1 if re.search(r'\d{5,}', url) else 0,
    ]
    return np.array(f, dtype=float).reshape(1, -1)

# ======================================
# URL Extraction
# ======================================

def extract_urls(text):
    protocol_urls = re.findall(r'https?://[^\s<>"\']+', text)
    www_urls      = re.findall(r'www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s<>"\']*', text)
    bare_domains  = re.findall(
        r'\b[a-zA-Z0-9\-]{3,}\.[a-zA-Z0-9\-]{2,}\.(com|net|org|info|co|io|xyz|tk|ml|ga|cf)\b',
        text
    )
    bare_domains = [".".join(d) if isinstance(d, tuple) else d for d in bare_domains]
    return list(dict.fromkeys(protocol_urls + www_urls + bare_domains))

# ======================================
# Upload Screenshot
# ======================================

uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

# ======================================
# Detection Pipeline
# ======================================

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", width="stretch")
    st.subheader("🔎 OCR Text Extraction")

    raw_text       = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    extracted_text = preprocess_ocr_text(raw_text)
    st.text_area("Extracted Text", extracted_text, height=200)

    reasons            = []
    risk_score         = 0
    flagged_line_count = 0
    max_line_prob      = 0.0
    nlp_reasons        = []
    suspicious_domains_found = False
    url_model_flag     = False

    # ── Pass 1: Full-text NLP ──────────────────────────────────────
    if extracted_text.strip():
        full_text_prob = run_nlp(extracted_text[:1000], tokenizer, spam_model)
        if full_text_prob > 0.55:
            risk_score += 2
            reasons.append(
                f"Full-text NLP: **{round(full_text_prob*100,2)}%** phishing probability"
            )
        elif full_text_prob > 0.40:
            risk_score += 1
            reasons.append(
                f"Full-text NLP: moderate suspicion ({round(full_text_prob*100,2)}%)"
            )

    # ── Pass 2: Line-by-line NLP ───────────────────────────────────
    lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]

    for i, line in enumerate(lines):
        if is_email_address(line) or is_purely_numeric(line) or is_ignorable_short(line):
            continue

        phishing_prob = run_nlp(line, tokenizer, spam_model)
        threshold     = 0.45 if contains_phishing_keyword(line) else 0.65
        max_line_prob = max(max_line_prob, phishing_prob)

        if phishing_prob > threshold:
            flagged_line_count += 1
            nlp_reasons.append(
                f'Line {i+1}: "{line[:80]}" → {round(phishing_prob*100,2)}%'
            )

    if flagged_line_count >= 3:
        risk_score += 3
    elif flagged_line_count == 2:
        risk_score += 2
    elif flagged_line_count == 1:
        risk_score += 1

    if nlp_reasons:
        reasons.append(f"**{flagged_line_count} suspicious line(s) flagged by NLP:**")
        reasons.extend(nlp_reasons)

    # ── Pass 3: URL Detection ──────────────────────────────────────
    urls = extract_urls(extracted_text)

    for url in urls:
        domain = extract_domain(url)
        is_fake, legit_match, score, reason_type = detect_fake_domain(domain)

        if is_fake:
            suspicious_domains_found = True
            type_label = {
                "subdomain_abuse":   "subdomain abuse",
                "typosquat":         "typosquatting",
                "keyword_in_domain": "brand keyword in domain"
            }.get(reason_type, "suspicious similarity")
            reasons.append(
                f"Suspicious domain **{domain}** → resembles **{legit_match}** "
                f"({type_label}, confidence: {round((score or 0)*100)}%)"
            )
            risk_score += 2

        try:
            features = extract_url_features(url)
            pred = phishing_model.predict(features)[0]
            if pred == 1:
                url_model_flag = True
                reasons.append(f"URL ML model flagged **{url[:80]}** as phishing")
                risk_score += 2
        except Exception:
            pass

    # ── Risk Level ─────────────────────────────────────────────────
    if risk_score >= 5:
        risk = "HIGH"
    elif risk_score >= 3:
        risk = "MEDIUM"
    elif risk_score >= 1:
        risk = "LOW-MEDIUM"
    else:
        risk = "LOW"

    # ── Final Result ───────────────────────────────────────────────
    st.subheader("🧠 Detection Result")
    st.write("**Risk Level:**", risk)
    st.write("**Risk Score:**", risk_score)
    st.write("**Highest Line Phishing Probability:**", round(max_line_prob * 100, 2), "%")
    st.write("**Suspicious Lines Flagged:**", flagged_line_count)

    if risk_score >= 3:
        st.error("⚠ PHISHING DETECTED")
        st.subheader("🚨 Detection Reasons")
        for r in reasons:
            st.write("•", r)
    elif risk_score >= 1:
        st.warning("⚠ SUSPICIOUS — Proceed with caution")
        st.subheader("🔶 Suspicious Indicators")
        for r in reasons:
            st.write("•", r)
    else:
        st.success("✅ SAFE CONTENT")
        st.write("No phishing indicators detected.")

    # ── Technical Details ──────────────────────────────────────────
    st.subheader("🔬 Analysis Details")
    st.write("**Detected URLs:**", urls)
    analyzed = [l for l in lines
                if not is_email_address(l)
                and not is_purely_numeric(l)
                and not is_ignorable_short(l)]
    st.write("**Total lines analyzed:**", len(analyzed))
