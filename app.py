from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import re
import socket
from urllib.parse import urlparse
from datetime import datetime

# ── Try to import optional live libraries ──────────────────────────────────
try:
    import whois                        # pip install python-whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

app = Flask(__name__)

# ── Feature columns (must match model.py) ─────────────────────────────────
FEATURE_COLS = [
    'URL_Length', 'Num_Dots', 'HTTPS', 'Domain_Age',
    'Num_Hyphens', 'Num_Slashes', 'Has_IP',
    'Suspicious_Keywords', 'Subdomain_Level', 'Has_At_Symbol'
]

SUSPICIOUS_WORDS = [
    'login', 'verify', 'update', 'secure', 'account',
    'banking', 'confirm', 'password', 'signin', 'ebayisapi',
    'webscr', 'paypal', 'appleid', 'support'
]

TRUSTED_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com',
    'amazon.in', 'wikipedia.org', 'microsoft.com', 'apple.com',
    'github.com', 'stackoverflow.com', 'chatgpt.com', 'openai.com',
    'linkedin.com', 'twitter.com', 'instagram.com', 'netflix.com',
    'flipkart.com', 'myntra.com', 'naukri.com', 'zomato.com',
]

# ── Load or train model ────────────────────────────────────────────────────
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        print("✅ Loaded saved model")
        return data['model'], data['accuracy']
    else:
        print("⚙️  No saved model found. Training now...")
        from model import train_and_save
        return train_and_save()

model, MODEL_ACCURACY = load_model()


# ── Feature Extraction ─────────────────────────────────────────────────────
def get_domain_age(domain):
    """Fetch real domain age via WHOIS (fallback = 1 year)."""
    if not WHOIS_AVAILABLE:
        return 1.0, "N/A (install python-whois)"
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age_years = (datetime.now() - creation).days / 365
            return round(age_years, 2), creation.strftime("%Y-%m-%d")
    except Exception:
        pass
    return 1.0, "Unknown"


def has_ip_address(url):
    """Check if URL uses an IP address instead of a domain name."""
    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
    return 1 if re.search(ip_pattern, url) else 0


def count_suspicious_keywords(url):
    url_lower = url.lower()
    return sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)


def get_subdomain_level(parsed):
    hostname = parsed.hostname or ''
    parts = hostname.split('.')
    # e.g. sub.sub.example.com → 4 parts → level = 4 - 2 = 2
    return max(0, len(parts) - 2)


def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    url_length   = len(url)
    num_dots     = url.count('.')
    https        = 1 if url.startswith('https') else 0
    num_hyphens  = url.count('-')
    num_slashes  = url.count('/')
    has_ip       = has_ip_address(url)
    sus_keywords = count_suspicious_keywords(url)
    subdomain_lvl= get_subdomain_level(parsed)
    has_at       = 1 if '@' in url else 0

    # Real domain age (or placeholder)
    domain_age, creation_date = get_domain_age(domain)

    features = {
        'URL_Length':          url_length,
        'Num_Dots':            num_dots,
        'HTTPS':               https,
        'Domain_Age':          domain_age,
        'Num_Hyphens':         num_hyphens,
        'Num_Slashes':         num_slashes,
        'Has_IP':              has_ip,
        'Suspicious_Keywords': sus_keywords,
        'Subdomain_Level':     subdomain_lvl,
        'Has_At_Symbol':       has_at,
    }
    return features, domain, creation_date


# ── Explainability ─────────────────────────────────────────────────────────
def explain(url, feats):
    reasons = []

    if feats['URL_Length'] > 54:
        reasons.append(f"⚠️ Very long URL ({feats['URL_Length']} chars) — common in phishing")

    if feats['Num_Dots'] > 3:
        reasons.append(f"⚠️ {feats['Num_Dots']} dots detected — multiple subdomains are suspicious")

    if feats['HTTPS'] == 0:
        reasons.append("⚠️ No HTTPS — connection is not encrypted")

    if feats['Domain_Age'] < 1:
        reasons.append(f"⚠️ Domain is very new ({feats['Domain_Age']} yrs) — phishing sites are often newly created")

    if feats['Num_Hyphens'] > 2:
        reasons.append(f"⚠️ {feats['Num_Hyphens']} hyphens in URL — often used to mimic trusted brands")

    if feats['Has_IP'] == 1:
        reasons.append("⚠️ URL uses an IP address instead of a domain name")

    if feats['Suspicious_Keywords'] > 0:
        found = [w for w in SUSPICIOUS_WORDS if w in url.lower()]
        reasons.append(f"⚠️ Suspicious keywords found: {', '.join(found)}")

    if feats['Subdomain_Level'] > 2:
        reasons.append(f"⚠️ Deep subdomain nesting (level {feats['Subdomain_Level']}) — classic phishing trick")

    if feats['Has_At_Symbol'] == 1:
        reasons.append("⚠️ '@' symbol in URL — browser ignores everything before it")

    if not reasons:
        reasons.append("✅ No major red flags detected in URL structure")

    return reasons


# ── Override Rules ─────────────────────────────────────────────────────────
def apply_overrides(url, result, confidence):
    """
    Hard rules that override the ML model for known safe/dangerous patterns.
    Returns (result, confidence, override_note)
    """
    parsed = urlparse(url)
    domain = (parsed.netloc or '').replace('www.', '')

    # Trusted domain check
    for td in TRUSTED_DOMAINS:
        if domain == td or domain.endswith('.' + td):
            return "Safe URL", 98.0, f"✅ Verified trusted domain: {td}"

    # Hard phishing signals
    if has_ip_address(url):
        return "Phishing URL", 97.0, "🚨 IP address used as domain — definite red flag"

    if '@' in url:
        return "Phishing URL", 95.0, "🚨 '@' symbol in URL redirects browser to attacker domain"

    return result, confidence, None


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', accuracy=MODEL_ACCURACY)


@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '').strip()

    if not url:
        return render_template('index.html', accuracy=MODEL_ACCURACY,
                               error="Please enter a URL.")

    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    # Extract features
    feats, domain, creation_date = extract_features(url)

    # ML Prediction + Confidence
    input_df   = pd.DataFrame([feats])[FEATURE_COLS]
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    confidence = round(float(max(proba)) * 100, 1)
    result     = "Safe URL" if prediction == 0 else "Phishing URL"

    # Apply override rules
    result, confidence, override_note = apply_overrides(url, result, confidence)

    # Explain
    reasons = explain(url, feats)

    return render_template(
        'index.html',
        accuracy=MODEL_ACCURACY,
        prediction_text=result,
        confidence=confidence,
        url_checked=url,
        domain=domain,
        creation_date=creation_date,
        url_length=feats['URL_Length'],
        num_dots=feats['Num_Dots'],
        https=feats['HTTPS'],
        domain_age=feats['Domain_Age'],
        num_hyphens=feats['Num_Hyphens'],
        num_slashes=feats['Num_Slashes'],
        has_ip=feats['Has_IP'],
        suspicious_keywords=feats['Suspicious_Keywords'],
        subdomain_level=feats['Subdomain_Level'],
        has_at=feats['Has_At_Symbol'],
        reasons=reasons,
        override_note=override_note,
    )


if __name__ == '__main__':
    app.run(debug=True)
