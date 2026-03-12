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
    div[class*="stError"] { background: rgba(255,60,90,0.08) !important; border: 1px solid var(--accent-red) !important; border-radius: 6px !important; color: var(--accent-red) !important; font-family: var(--font-display) !important; font-size: 1.05rem !important; font-weight: 700 !important; }
    div[class*="stWarning"] { background: rgba(255,179,0,0.08) !important; border: 1px solid var(--accent-amber) !important; border-radius: 6px !important; color: var(--accent-amber) !important; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-dark); } ::-webkit-scrollbar-thumb { background: #1e2b3a; border-radius: 3px; } ::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
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
# Legitimate Domains — expanded
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
# WHY: Raw OCR contains noise characters, broken spacing, and unicode
#      look-alikes (e.g. Cyrillic 'а' instead of Latin 'a') that
#      degrade NLP model confidence. Cleaning first improves accuracy.
# ======================================

def preprocess_ocr_text(text):
    # Normalize unicode homoglyphs attackers use to evade filters
    homoglyphs = {"ρ":"p","а":"a","е":"e","о":"o","і":"i","с":"c",
                  "ν":"v","μ":"u","η":"n","τ":"t","κ":"k","ζ":"z"}
    for fake, real in homoglyphs.items():
        text = text.replace(fake, real)
    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are pure noise
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 1:
            continue
        # Skip lines that are >70% non-alphanumeric (OCR garbage)
        alnum = sum(c.isalnum() or c.isspace() for c in stripped)
        if alnum / max(len(stripped), 1) < 0.3:
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)

# ======================================
# Phishing Keywords — expanded from 11 to 50+
# WHY: Original list missed common lures like "otp", "kyc", "expire",
#      brand names, and action verbs used in phishing copy.
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
# Revised Line Filters
# WHY — original bugs:
#   is_email: matched mid-line emails inside sentences
#   mostly_numbers: dropped "Your OTP is 482910 — verify now"
#   too_short: dropped "Verify your account" (3 words) and "Login" button text
# ======================================

def is_email_address(line):
    # Only skip lines that ARE an email address, not lines containing one
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", line.strip()))

def is_purely_numeric(line):
    # Skip only if zero alphabetic words exist
    tokens = line.split()
    word_count = sum(1 for t in tokens if re.search(r'[a-zA-Z]', t))
    return word_count == 0

def is_ignorable_short(line):
    # Only skip 1-token lines that aren't phishing keywords
    tokens = line.split()
    if len(tokens) >= 2:
        return False
    return not contains_phishing_keyword(line)

# ======================================
# NLP Runner
# ======================================

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
        return probs[0][1].item()

# ======================================
# Domain Extraction — improved
# WHY: Original split only on "/" leaving port numbers and query
#      strings attached, causing similarity checks to fail.
# ======================================

def extract_domain(url):
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    domain = re.split(r'[/:?#]', url)[0].lower().strip()
    return domain

# ======================================
# Homoglyph Normalization for domain comparison
# WHY: 'paypa1.com' scores only ~0.77 similarity vs threshold 0.85
#      After normalizing '1'→'l', it becomes 'paypal.com' → caught.
# ======================================

def normalize_homoglyphs(s):
    return (s.replace("0","o").replace("1","l").replace("3","e")
             .replace("4","a").replace("5","s").replace("@","a")
             .replace("rn","m").replace("vv","w"))

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ======================================
# Fake Domain Detection — improved
# WHY:
#   Threshold lowered 0.85 → 0.72 to catch typosquats
#   Added homoglyph normalization before comparison
#   Added subdomain abuse detection (paypal.com.evil.ru)
# ======================================

def detect_fake_domain(domain):
    norm_domain = normalize_homoglyphs(domain)

    for legit in legit_domains:
        legit_base = legit.split(".")[0]
        norm_legit  = normalize_homoglyphs(legit)

        # Subdomain abuse: "paypal.com.attacker.ru"
        if legit in domain and not domain.endswith(legit):
            return True, legit, 1.0, "subdomain_abuse"

        # Typosquat with homoglyph-normalized comparison
        score = similarity(norm_domain, norm_legit)
        if score >= 0.72 and domain != legit:
            return True, legit, score, "typosquat"

        # Brand keyword embedded in domain
        if legit_base in domain and domain != legit:
            return True, legit, score, "keyword_in_domain"

    return False, None, None, None

# ======================================
# URL Feature Extraction — expanded 5 → 14 features
# WHY: Original 5 features missed IP-based URLs, deep paths,
#      URL encoding obfuscation, and phishing keywords in path.
# ======================================

def extract_url_features(url):
    f = []
    f.append(len(url))                                                              # f1: length
    f.append(url.count("."))                                                        # f2: dots
    f.append(url.count("-"))                                                        # f3: hyphens
    f.append(1 if "@" in url else 0)                                                # f4: @ symbol
    f.append(1 if url.startswith("https") else 0)                                   # f5: https
    f.append(url.count("/"))                                                        # f6: slashes (path depth)
    f.append(url.count("="))                                                        # f7: query params
    f.append(url.count("?"))                                                        # f8: query string
    f.append(url.count("_"))                                                        # f9: underscores
    f.append(1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0)   # f10: IP address
    f.append(len(url.split("/")[0]) if "/" in url else len(url))                    # f11: domain length
    f.append(1 if re.search(r'(secure|login|verify|update|account|banking)', url.lower()) else 0)  # f12: keywords in URL
    f.append(url.count("%"))                                                        # f13: URL encoding
    f.append(1 if re.search(r'\d{5,}', url) else 0)                                # f14: long numeric sequence
    return np.array(f).reshape(1, -1)

# ======================================
# URL Extraction — improved
# WHY: Original missed bare domains without http:// prefix which are
#      common in image-based phishing (e.g. "visit paypa1-secure.com")
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

    # WHY --oem 3 --psm 6: oem 3 = best available engine (LSTM),
    # psm 6 = assume a uniform block of text — better for email/web screenshots
    raw_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    extracted_text = preprocess_ocr_text(raw_text)

    st.text_area("Extracted Text", extracted_text, height=200)

    reasons        = []
    risk_score     = 0
    flagged_line_count = 0
    max_line_prob  = 0.0
    nlp_reasons    = []
    suspicious_domains_found = False
    url_model_flag = False

    # ======================================
    # Pass 1: Full-text NLP
    # WHY: Line-by-line NLP misses context that spans multiple lines.
    # Running the whole OCR block as one input catches holistic phishing
    # tone that individual lines don't carry alone.
    # ======================================

    if extracted_text.strip():
        full_text_prob = run_nlp(extracted_text[:1000], tokenizer, spam_model)
        if full_text_prob > 0.55:
            risk_score += 2
            reasons.append(
                f"Full-text NLP: **{round(full_text_prob*100,2)}%** phishing probability across entire content"
            )
        elif full_text_prob > 0.40:
            risk_score += 1
            reasons.append(
                f"Full-text NLP: moderate suspicion ({round(full_text_prob*100,2)}%)"
            )

    # ======================================
    # Pass 2: Line-by-line NLP
    # WHY: Uses revised filters so short keyword-lines and OTP-lure lines
    #      are no longer silently dropped. Dynamic threshold means lines
    #      without keywords need higher confidence to flag (reduces FP).
    # ======================================

    lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]

    for i, line in enumerate(lines):
        if is_email_address(line):
            continue
        if is_purely_numeric(line):
            continue
        if is_ignorable_short(line):
            continue

        phishing_prob = run_nlp(line, tokenizer, spam_model)
        # Lines with phishing keywords: lower bar (0.45)
        # Lines without keywords: higher bar (0.65) to reduce false positives
        threshold = 0.45 if contains_phishing_keyword(line) else 0.65

        max_line_prob = max(max_line_prob, phishing_prob)

        if phishing_prob > threshold:
            flagged_line_count += 1
            nlp_reasons.append(
                f'Line {i+1}: "{line[:80]}" → {round(phishing_prob*100,2)}%'
            )

    # WHY: Count-based scoring — 3 flagged lines is stronger evidence
    # than 1 borderline line, even if max_prob is similar.
    if flagged_line_count >= 3:
        risk_score += 3
    elif flagged_line_count == 2:
        risk_score += 2
    elif flagged_line_count == 1:
        risk_score += 1

    if nlp_reasons:
        reasons.append(f"**{flagged_line_count} suspicious line(s) flagged by NLP:**")
        reasons.extend(nlp_reasons)

    # ======================================
    # Pass 3: URL Detection
    # ======================================

    urls = extract_urls(extracted_text)

    for url in urls:
        domain = extract_domain(url)

        is_fake, legit_match, score, reason_type = detect_fake_domain(domain)

        if is_fake:
            suspicious_domains_found = True
            type_label = {
                "subdomain_abuse":    "subdomain abuse",
                "typosquat":          "typosquatting",
                "keyword_in_domain":  "brand keyword in domain"
            }.get(reason_type, "suspicious similarity")
            reasons.append(
                f"Suspicious domain **{domain}** → resembles **{legit_match}** "
                f"({type_label}, confidence: {round((score or 0)*100)}%)"
            )
            risk_score += 2

        try:
            features = extract_url_features(url)
            # Graceful fallback: if saved model was trained on 5 features
            try:
                pred = phishing_model.predict(features)[0]
            except Exception:
                pred = phishing_model.predict(features[:, :5])[0]

            if pred == 1:
                url_model_flag = True
                reasons.append(f"URL ML model flagged **{url[:80]}** as phishing")
                risk_score += 2

        except Exception:
            pass

    # ======================================
    # Risk Level
    # WHY: Thresholds rescaled for new max possible score (~15+).
    # Added LOW-MEDIUM tier so borderline cases don't silently pass as SAFE.
    # ======================================

    if risk_score >= 5:
        risk = "HIGH"
    elif risk_score >= 3:
        risk = "MEDIUM"
    elif risk_score >= 1:
        risk = "LOW-MEDIUM"
    else:
        risk = "LOW"

    # ======================================
    # Final Result
    # ======================================

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

    # ======================================
    # Technical Details
    # ======================================

    st.subheader("🔬 Analysis Details")
    st.write("**Detected URLs:**", urls)
    analyzed_lines = [
        l for l in lines
        if not is_email_address(l) and not is_purely_numeric(l) and not is_ignorable_short(l)
    ]
    st.write("**Total lines analyzed:**", len(analyzed_lines))
