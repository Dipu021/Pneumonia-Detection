"""
PneumoScan AI — Pneumonia Detection System
==========================================
Features:
  1. ResNet-18 chest X-ray classification (Normal / Pneumonia)
  2. Do's & Don'ts and emergency signs
  3. Nearest hospital & doctor finder (Google Maps deep-link + sample cards)
  4. Professional clinical dark-theme UI (Syne + IBM Plex Mono + Inter)

Requirements:
  pip install streamlit torch torchvision pillow

Model:
  Place your trained weights at:  model/pneumonia_model.pth
  Architecture: ResNet-18, 2-class output (Normal=0, Pneumonia=1)
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import time

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI · Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — clinical dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0b0f1a; color:#d0d8f0; }
#MainMenu, footer, header   { visibility:hidden; }
.block-container             { padding:2rem 3rem 4rem; max-width:1300px; }

/* ── sidebar ── */
[data-testid="stSidebar"]   { background:#0d1220; border-right:1px solid #1e2a45; }
[data-testid="stSidebar"] * { color:#a0aec0 !important; }
[data-testid="stSidebar"] hr{ border-color:#1e2a45; }

/* ── hero ── */
.hero-wrap { background:linear-gradient(135deg,#0f1e3d 0%,#0b1628 60%,#0d1f1a 100%);
    border:1px solid #1e3a5f; border-radius:16px; padding:2.5rem 3rem;
    margin-bottom:2rem; position:relative; overflow:hidden; }
.hero-wrap::before { content:''; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(96,165,250,.12) 0%,transparent 70%);
    border-radius:50%; pointer-events:none; }
.hero-title { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
    color:#e2eaf8; letter-spacing:-1px; margin:0; line-height:1.1; }
.hero-title span { color:#60a5fa; }
.hero-sub   { font-size:.95rem; color:#6b7fa3; margin-top:.6rem; font-weight:300; letter-spacing:.5px; }
.hero-badge { display:inline-block; background:rgba(96,165,250,.12);
    border:1px solid rgba(96,165,250,.25); color:#60a5fa;
    font-family:'IBM Plex Mono',monospace; font-size:.72rem;
    padding:3px 10px; border-radius:4px; margin-bottom:1rem; letter-spacing:1px; }

/* ── upload zone ── */
[data-testid="stFileUploader"]        { background:#0e1627 !important; border:1.5px dashed #2a3f6f !important; border-radius:12px !important; }
[data-testid="stFileUploader"]:hover  { border-color:#60a5fa !important; }
[data-testid="stFileUploadDropzone"]  { background:transparent !important; }

/* ── generic card & title ── */
.card       { background:#0e1627; border:1px solid #1e2d50; border-radius:14px; padding:1.5rem; }
.card-title { font-family:'Syne',sans-serif; font-size:.8rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; color:#4a6fa5;
    margin-bottom:1rem; display:flex; align-items:center; gap:.5rem; }

/* ── result banners ── */
.result-pneumonia { background:linear-gradient(135deg,#2a0d0d,#1a0606);
    border:1px solid #7f1d1d; border-left:4px solid #ef4444;
    border-radius:10px; padding:1.2rem 1.5rem; margin:1rem 0; }
.result-normal    { background:linear-gradient(135deg,#0d2a1a,#061a10);
    border:1px solid #14532d; border-left:4px solid #22c55e;
    border-radius:10px; padding:1.2rem 1.5rem; margin:1rem 0; }
.result-label     { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; margin:0; }
.result-pneumonia .result-label { color:#f87171; }
.result-normal    .result-label { color:#4ade80; }
.result-sub { font-size:.82rem; color:#6b7fa3; margin-top:4px; }

/* ── confidence bar ── */
.conf-bar-track { background:#1a2540; border-radius:99px; height:10px;
    width:100%; margin:.5rem 0 .3rem; overflow:hidden; }
.conf-bar-fill  { height:100%; border-radius:99px; }
.conf-high { background:linear-gradient(90deg,#22c55e,#4ade80); }
.conf-mid  { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.conf-low  { background:linear-gradient(90deg,#ef4444,#f87171); }
.conf-label{ font-family:'IBM Plex Mono',monospace; font-size:.78rem; color:#6b7fa3;
    display:flex; justify-content:space-between; }

/* ── probability rows ── */
.prob-row  { display:flex; justify-content:space-between; align-items:center;
    padding:.6rem 0; border-bottom:1px solid #1a2540; }
.prob-row:last-child { border-bottom:none; }
.prob-name { font-size:.88rem; color:#a0aec0; display:flex; align-items:center; gap:.5rem; }
.prob-val  { font-family:'IBM Plex Mono',monospace; font-size:.9rem; font-weight:500; color:#e2eaf8; }

/* ── stat chips ── */
.stat-chip     { background:#131e35; border:1px solid #1e2d50; border-radius:10px;
    padding:.9rem 1.2rem; text-align:center; }
.stat-chip .val{ font-family:'IBM Plex Mono',monospace; font-size:1.25rem; font-weight:500; color:#60a5fa; }
.stat-chip .lbl{ font-size:.72rem; color:#4a6fa5; letter-spacing:1px; text-transform:uppercase; margin-top:3px; }

/* ── info grids ── */
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:.8rem; }
.info-item { background:#131e35; border-radius:8px; padding:.7rem 1rem; font-size:.82rem; }
.info-item .key { color:#4a6fa5; margin-bottom:2px; font-size:.72rem; letter-spacing:.5px; }
.info-item .val { color:#d0d8f0; font-weight:500; }

/* ── disclaimer ── */
.disclaimer { background:#120f05; border:1px solid #3d2b00; border-radius:10px;
    padding:.9rem 1.2rem; font-size:.8rem; color:#a0803a;
    display:flex; gap:.7rem; align-items:flex-start; margin-top:1rem; }

/* ── divider ── */
.divider { border:none; border-top:1px solid #1e2d50; margin:1.5rem 0; }

/* ── section header ── */
.section-header { font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:800;
    color:#e2eaf8; letter-spacing:-.3px; margin:2rem 0 1rem;
    display:flex; align-items:center; gap:.6rem; }

/* ── do / don't ── */
.do-dont-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:.8rem; }
.do-card      { background:#0a1f14; border:1px solid #14532d; border-radius:12px; padding:1rem; }
.dont-card    { background:#1a0606;  border:1px solid #7f1d1d; border-radius:12px; padding:1rem; }
.do-card h4   { color:#4ade80; font-size:.82rem; margin:0 0 .6rem; letter-spacing:1px; }
.dont-card h4 { color:#f87171; font-size:.82rem; margin:0 0 .6rem; letter-spacing:1px; }
.do-item, .dont-item { font-size:.8rem; color:#a0aec0; padding:3px 0; display:flex; gap:6px; }

/* ── hospital cards ── */
.hosp-card { background:#0e1627; border:1px solid #1e2d50; border-radius:12px;
    padding:1rem 1.2rem; margin-bottom:.7rem; display:flex; gap:1rem; align-items:flex-start; }
.hosp-icon { font-size:1.6rem; flex-shrink:0; }
.hosp-name { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:700; color:#e2eaf8; }
.hosp-dist { font-family:'IBM Plex Mono',monospace; font-size:.75rem; color:#60a5fa; }
.hosp-spec { display:inline-block; font-size:.68rem; padding:2px 8px; border-radius:4px;
    background:rgba(96,165,250,.12); color:#60a5fa; margin-top:3px; }
.hosp-addr { font-size:.78rem; color:#6b7fa3; margin-top:3px; }
.hosp-link { font-size:.75rem; color:#60a5fa; text-decoration:none; }

/* ── session log table ── */
.log-table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:.6rem; }
.log-table th { color:#4a6fa5; font-weight:600; text-align:left; padding:.5rem .8rem;
    border-bottom:1px solid #1e2d50; letter-spacing:.5px; font-size:.72rem; text-transform:uppercase; }
.log-table td { padding:.5rem .8rem; border-bottom:1px solid #131e35; color:#a0aec0; }
.log-table tr:last-child td { border-bottom:none; }
.badge-pneu  { background:rgba(239,68,68,.15);  color:#f87171; padding:2px 8px; border-radius:4px; font-size:.72rem; }
.badge-norm  { background:rgba(34,197,94,.15);  color:#4ade80; padding:2px 8px; border-radius:4px; font-size:.72rem; }

/* ── footer ── */
.footer { text-align:center; font-size:.75rem; color:#2a3f6f;
    margin-top:3rem; font-family:'IBM Plex Mono',monospace; letter-spacing:.5px; }

/* ── misc ── */
label, .stFileUploader label { color:#6b7fa3 !important; }
.stTextInput input { background:#0e1627 !important; border:1px solid #1e2d50 !important;
    color:#d0d8f0 !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA — guidance, hospitals
# ─────────────────────────────────────────────────────────────────────────────
PNEUMONIA_GUIDANCE = {
    "dos": [
        "Rest completely — your immune system needs energy to fight infection",
        "Drink 8–10 glasses of water or clear warm fluids daily",
        "Complete the full antibiotic course even if you feel better",
        "Sleep with your head elevated (2–3 pillows) to ease breathing",
        "Monitor oxygen with a pulse oximeter — target SpO₂ ≥ 95%",
        "Eat nutrient-rich foods: soups, fruits, vegetables, lean proteins",
        "Take deep breathing exercises every 2 hours while awake",
        "Seek emergency care immediately if SpO₂ drops below 92%",
    ],
    "donts": [
        "Do NOT stop antibiotics early — leads to drug resistance",
        "Avoid smoking or secondhand smoke completely",
        "Do NOT take cough suppressants with a productive cough",
        "Avoid alcohol — it weakens immunity and interacts with drugs",
        "Do not delay hospital visit if breathing worsens rapidly",
        "Avoid cold or very cold environments and wind",
        "Do not self-prescribe antibiotics without a physician",
        "Avoid strenuous physical activity during recovery",
    ],
    "emergency_signs": [
        "Lips, fingernails, or face turning blue (cyanosis)",
        "Breathing faster than 30 breaths per minute",
        "Severe chest pain, confusion, or loss of consciousness",
        "SpO₂ below 92% on pulse oximeter",
        "High fever (>39.4°C / 103°F) unresponsive to paracetamol",
        "Coughing up blood or blood-streaked sputum",
    ],
    "diet_tips": [
        ("🍵", "Warm ginger-honey tea", "Soothes airways, anti-inflammatory"),
        ("🍗", "Chicken broth / soup",   "Hydration + electrolytes + easy nutrition"),
        ("🍊", "Vitamin C foods",         "Oranges, kiwi, guava — boost immune response"),
        ("🧄", "Garlic",                  "Natural antimicrobial properties"),
        ("🥦", "Leafy greens",            "Vitamins A, C, K — support lung tissue repair"),
        ("💧", "Warm water + lemon",      "Stays hydrated, loosens mucus secretions"),
    ],
}

NORMAL_GUIDANCE = {
    "tips": [
        "Your scan shows no pneumonia — keep maintaining good lung health with regular exercise.",
        "Wash hands frequently (20 sec with soap) to prevent respiratory infections.",
        "Get annual influenza and pneumococcal vaccines.",
        "Stay hydrated and avoid prolonged exposure to dust, smoke, or air pollution.",
        "If you still have symptoms despite a normal scan, consult a physician — clinical examination is still crucial.",
        "Practice deep breathing exercises or yoga to keep lungs strong.",
    ]
}

SAMPLE_HOSPITALS = [
    {"name":"City Chest & Pulmonary Centre",     "spec":"Pulmonologist · ICU",         "dist":"0.8 km","addr":"Central Road, near Bus Stand","icon":"🏥","phone":"112"},
    {"name":"Apollo / Fortis Affiliated Clinic", "spec":"General Physician · Radiology","dist":"1.4 km","addr":"Main Market Road",            "icon":"🏨","phone":"104"},
    {"name":"Government General Hospital",       "spec":"Emergency · All Departments",  "dist":"2.1 km","addr":"District Headquarters Area",  "icon":"🏛","phone":"108"},
    {"name":"Sunrise Respiratory Clinic",        "spec":"Chest Specialist · Spirometry","dist":"3.2 km","addr":"New Colony, Sector 4",        "icon":"🏪","phone":"102"},
]

SPECIALTIES = ["Pulmonologist","General Physician","Chest Specialist","Emergency / ICU","Radiologist"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;color:#60a5fa;">🫁 PneumoScan AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**SYSTEM STATUS**")
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:1.2rem;">
        <div style="display:flex;align-items:center;gap:8px;font-size:.82rem;"><span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;"></span>Model Loaded</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:.82rem;"><span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;"></span>ResNet-18 Active</div>
        <div style="display:flex;align-items:center;gap:8px;font-size:.82rem;"><span style="width:8px;height:8px;border-radius:50%;background:#f59e0b;display:inline-block;"></span>CPU Inference</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**MODEL INFO**")
    st.markdown("""
    <div style="font-size:.8rem;line-height:2;color:#6b7fa3;">
        Architecture &nbsp;&nbsp;<span style="color:#a0aec0;">ResNet-18</span><br>
        Classes &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#a0aec0;">Normal · Pneumonia</span><br>
        Input size &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#a0aec0;">224 × 224 px</span><br>
        Normalisation &nbsp;<span style="color:#a0aec0;">ImageNet mean/std</span><br>
        Framework &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#a0aec0;">PyTorch</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**HOW TO USE**")
    st.markdown("""
    <ol style="font-size:.8rem;color:#6b7fa3;padding-left:1.2rem;line-height:2.4;">
        <li>Upload a chest X-ray (JPG / PNG)</li>
        <li>Review AI prediction &amp; confidence</li>
        <li>Read care guidance &amp; Do's / Don'ts</li>
        <li>Check diet recommendations</li>
        <li>Find nearest hospitals</li>
        <li>Always consult a physician</li>
    </ol>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if "scan_count" not in st.session_state:
        st.session_state.scan_count = 0
    st.markdown(f"""
    <div style="font-size:.8rem;color:#6b7fa3;line-height:2;">
        Session scans &nbsp;<span style="color:#60a5fa;font-family:'IBM Plex Mono',monospace;">{st.session_state.scan_count}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.72rem;color:#3d4f70;line-height:1.6;">
        ⚠️ For <strong style="color:#5a6f94;">educational use only</strong>.
        Not a certified medical device. Always seek professional clinical evaluation.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = models.resnet18(weights=None)
    mdl.fc = nn.Linear(mdl.fc.in_features, 2)
    mdl.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=dev))
    mdl.to(dev).eval()
    return mdl, dev

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">RADIOLOGY · AI-ASSISTED · v3.0</div>
    <h1 class="hero-title">Chest X-Ray <span>Pneumonia</span><br>Detection System</h1>
    <p class="hero-sub">
        AI diagnosis &nbsp;·&nbsp; Care guidance &nbsp;·&nbsp; Diet recommendations
        &nbsp;·&nbsp; Hospital finder
    </p>
</div>
""", unsafe_allow_html=True)

# ── top stat chips ──
c1, c2, c3, c4 = st.columns(4)
for col, (val, lbl) in zip([c1, c2, c3, c4], [
    ("ResNet-18",  "Architecture"),
    ("224 × 224",  "Input Size"),
    ("🥗 Diet Tips", "Nutrition Guide"),
    ("🏥 Locator",  "Hospital Finder"),
]):
    with col:
        st.markdown(f'<div class="stat-chip"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Syne',sans-serif;font-size:.75rem;font-weight:700;
    letter-spacing:2px;text-transform:uppercase;color:#4a6fa5;margin-bottom:.6rem;">
    📤 &nbsp;UPLOAD CHEST X-RAY IMAGE
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop or click to browse — JPG, JPEG, PNG supported",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FLOW — runs only after image is uploaded
# ═════════════════════════════════════════════════════════════════════════════
if uploaded_file:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── inference ──────────────────────────────────────────────────────────
    image      = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    t_start = time.time()
    with st.spinner("🔬 Analysing chest X-ray..."):
        time.sleep(0.3)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs   = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
    infer_ms = round((time.time() - t_start) * 1000)

    CLASSES         = ["Normal", "Pneumonia"]
    result          = CLASSES[pred.item()]
    conf_score      = conf.item() * 100
    prob_normal     = round(float(probs[0][0]) * 100, 2)
    prob_pneumonia  = round(float(probs[0][1]) * 100, 2)
    bar_cls         = "conf-high" if conf_score >= 80 else ("conf-mid" if conf_score >= 60 else "conf-low")

    st.session_state.scan_count += 1

    if "scan_log" not in st.session_state:
        st.session_state.scan_log = []
    st.session_state.scan_log.append({
        "file":   uploaded_file.name,
        "result": result,
        "conf":   f"{conf_score:.1f}%",
        "ms":     f"{infer_ms} ms",
    })

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — SCAN RESULT
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🫁 &nbsp;Scan Analysis</div>', unsafe_allow_html=True)

    col_img, col_rep = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="card-title">🖼 &nbsp;INPUT IMAGE</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=f"File: {uploaded_file.name}")
        w, h = image.size
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-item"><div class="key">FILENAME</div><div class="val">{uploaded_file.name}</div></div>
            <div class="info-item"><div class="key">DIMENSIONS</div><div class="val">{w} × {h} px</div></div>
            <div class="info-item"><div class="key">FORMAT</div><div class="val">{uploaded_file.type.split("/")[-1].upper()}</div></div>
            <div class="info-item"><div class="key">INFER TIME</div><div class="val">{infer_ms} ms</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_rep:
        st.markdown('<div class="card-title">📋 &nbsp;PREDICTION REPORT</div>', unsafe_allow_html=True)

        if result == "Pneumonia":
            st.markdown("""
            <div class="result-pneumonia">
                <p class="result-label">🚨 &nbsp;Pneumonia Detected</p>
                <p class="result-sub">Abnormal radiological pattern identified. Clinical evaluation strongly recommended.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <p class="result-label">✅ &nbsp;Normal — No Pathology Found</p>
                <p class="result-sub">No significant radiological abnormality detected in this scan.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1.2rem;">
            <div class="card-title">📊 &nbsp;CONFIDENCE SCORE</div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill {bar_cls}" style="width:{conf_score:.1f}%;"></div>
            </div>
            <div class="conf-label">
                <span>0%</span>
                <span style="color:#e2eaf8;font-weight:500;">{conf_score:.2f}%</span>
                <span>100%</span>
            </div>
        </div>

        <div style="margin-top:1.4rem;">
            <div class="card-title">🧬 &nbsp;CLASS PROBABILITIES</div>
        </div>
        <div style="background:#0e1627;border:1px solid #1e2d50;border-radius:10px;padding:.5rem 1rem;">
            <div class="prob-row">
                <div class="prob-name">
                    <span style="width:10px;height:10px;border-radius:50%;background:#22c55e;display:inline-block;"></span>
                    Normal
                </div>
                <div class="prob-val">{prob_normal:.2f}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-name">
                    <span style="width:10px;height:10px;border-radius:50%;background:#ef4444;display:inline-block;"></span>
                    Pneumonia
                </div>
                <div class="prob-val">{prob_pneumonia:.2f}%</div>
            </div>
        </div>

        <div class="info-grid">
            <div class="info-item"><div class="key">PREDICTION</div><div class="val">{result}</div></div>
            <div class="info-item"><div class="key">CONFIDENCE</div><div class="val">{conf_score:.1f}%</div></div>
            <div class="info-item"><div class="key">DEVICE</div><div class="val">{"GPU · CUDA" if device.type=="cuda" else "CPU"}</div></div>
            <div class="info-item"><div class="key">MODEL</div><div class="val">ResNet-18</div></div>
        </div>
        <div class="disclaimer">
            <span>⚠️</span>
            <span>This AI output is <strong>not a medical diagnosis</strong>.
            Always consult a licensed radiologist or physician before any clinical decision.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── session log ────────────────────────────────────────────────────────
    if len(st.session_state.scan_log) > 1:
        with st.expander("📋  Session History — all scans this session", expanded=False):
            rows = "".join(
                f"""<tr>
                    <td>{i+1}</td>
                    <td style="color:#d0d8f0;">{s['file']}</td>
                    <td><span class="{'badge-pneu' if s['result']=='Pneumonia' else 'badge-norm'}">{s['result']}</span></td>
                    <td>{s['conf']}</td>
                    <td>{s['ms']}</td>
                </tr>"""
                for i, s in enumerate(st.session_state.scan_log)
            )
            st.markdown(f"""
            <table class="log-table">
                <thead><tr><th>#</th><th>File</th><th>Result</th><th>Confidence</th><th>Infer Time</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — CARE GUIDANCE (Do's, Don'ts, Emergency)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🩺 &nbsp;Care Guidance</div>', unsafe_allow_html=True)

    if result == "Pneumonia":
        # Do's & Don'ts
        st.markdown('<div class="card-title">✅ &nbsp;DO\'S &amp; ❌ DON\'TS</div>', unsafe_allow_html=True)
        dos_html   = "".join(f'<div class="do-item"><span>✔</span><span>{d}</span></div>'   for d in PNEUMONIA_GUIDANCE["dos"])
        donts_html = "".join(f'<div class="dont-item"><span>✖</span><span>{d}</span></div>' for d in PNEUMONIA_GUIDANCE["donts"])
        st.markdown(f"""
        <div class="do-dont-grid">
            <div class="do-card"><h4>✅ &nbsp;DO THESE</h4>{dos_html}</div>
            <div class="dont-card"><h4>❌ &nbsp;AVOID THESE</h4>{donts_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Emergency signs
        signs_html = "".join(
            f'<div style="display:flex;gap:8px;padding:5px 0;font-size:.8rem;color:#fca5a5;border-bottom:1px solid #3a1010;">'
            f'<span>🚨</span><span>{s}</span></div>'
            for s in PNEUMONIA_GUIDANCE["emergency_signs"]
        )
        st.markdown(f"""
        <div style="background:#1a0606;border:1px solid #7f1d1d;border-radius:12px;padding:1rem 1.2rem;margin-top:1.2rem;">
            <div style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                color:#f87171;letter-spacing:1px;margin-bottom:.6rem;">
                🚨 &nbsp;EMERGENCY — CALL 112 / GO TO ER IMMEDIATELY IF YOU NOTICE:
            </div>
            {signs_html}
        </div>
        """, unsafe_allow_html=True)

    else:
        tips_html = "".join(
            f'<div style="display:flex;gap:8px;padding:6px 0;font-size:.83rem;color:#a0aec0;border-bottom:1px solid #1e2d50;">'
            f'<span style="color:#4ade80;">✔</span><span>{t}</span></div>'
            for t in NORMAL_GUIDANCE["tips"]
        )
        st.markdown(f"""
        <div style="background:#0a1f14;border:1px solid #14532d;border-radius:12px;padding:1.2rem 1.5rem;">
            <div class="card-title">🌿 &nbsp;LUNG HEALTH TIPS</div>
            {tips_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — DIET & NUTRITION RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🥗 &nbsp;Diet & Nutrition Recommendations</div>', unsafe_allow_html=True)

    if result == "Pneumonia":
        st.markdown("""
        <div style="font-size:.82rem;color:#6b7fa3;margin-bottom:1rem;">
            A nutrient-rich diet speeds recovery and supports your immune system. Focus on warm, easy-to-digest foods.
        </div>
        """, unsafe_allow_html=True)

        diet_cols = st.columns(3)
        for i, (icon, food, tip) in enumerate(PNEUMONIA_GUIDANCE["diet_tips"]):
            diet_cols[i % 3].markdown(f"""
            <div style="background:#0e1627;border:1px solid #1e2d50;border-radius:12px;
                padding:.9rem 1rem;margin-bottom:.8rem;text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:.88rem;font-weight:700;
                    color:#93c5fd;margin:.4rem 0 .2rem;">{food}</div>
                <div style="font-size:.75rem;color:#6b7fa3;">{tip}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0a1020;border:1px solid #1e3a5f;border-radius:10px;
            padding:.9rem 1.2rem;margin-top:.4rem;font-size:.82rem;color:#6b7fa3;display:flex;gap:.8rem;align-items:center;">
            <span style="font-size:1.4rem;">💧</span>
            <span><strong style="color:#93c5fd;">Hydration goal:</strong>
            Drink at least <strong style="color:#60a5fa;">8–10 glasses</strong> of warm water, herbal teas,
            or clear broths per day. Avoid cold drinks, caffeine, and alcohol.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0a1f14;border:1px solid #14532d;border-radius:12px;padding:1rem 1.4rem;font-size:.83rem;color:#a0aec0;">
            <strong style="color:#4ade80;">🌿 Healthy lungs need:</strong> antioxidant-rich foods (berries, leafy greens),
            lean proteins, omega-3 fatty acids (fish, walnuts), and staying well-hydrated.
            Avoid processed foods, excessive sugar, and smoking.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — NEAREST HOSPITALS & DOCTORS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🏥 &nbsp;Find Nearest Hospitals & Doctors</div>', unsafe_allow_html=True)

    col_loc1, col_loc2, col_loc3 = st.columns([3, 2, 1])
    with col_loc1:
        city_input = st.text_input(
            "📍 Enter your city or area",
            placeholder="e.g. Kathmandu, Mumbai, London, New York...",
        )
    with col_loc2:
        specialty = st.selectbox("Specialty needed", SPECIALTIES)
    with col_loc3:
        st.markdown("<br>", unsafe_allow_html=True)
        search_maps = st.button("🗺 Search Maps", use_container_width=True)

    if city_input:
        query    = f"{specialty} hospital near {city_input}".replace(" ", "+")
        maps_url = f"https://www.google.com/maps/search/{query}"

        st.markdown(f"""
        <div style="background:#0e1627;border:1px solid #1e2d50;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
            <div class="card-title">🗺 &nbsp;MAPS SEARCH — {city_input.upper()}</div>
            <p style="font-size:.85rem;color:#6b7fa3;margin-bottom:1rem;">
                Search for <strong style="color:#93c5fd;">{specialty}s</strong> near
                <strong style="color:#93c5fd;">{city_input}</strong> in real time.
            </p>
            <div style="display:flex;gap:.8rem;flex-wrap:wrap;">
                <a href="{maps_url}" target="_blank" style="display:inline-block;background:#1e3a5f;color:#93c5fd;
                    font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
                    padding:.6rem 1.4rem;border-radius:8px;text-decoration:none;border:1px solid #2a5a8a;">
                    🗺 &nbsp;Google Maps →
                </a>
                <a href="https://www.google.com/maps/search/hospital+emergency+{city_input.replace(' ','+')}"
                    target="_blank" style="display:inline-block;background:#2a0d0d;color:#f87171;
                    font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
                    padding:.6rem 1.4rem;border-radius:8px;text-decoration:none;border:1px solid #7f1d1d;">
                    🚨 &nbsp;Emergency Hospitals →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title">📋 &nbsp;NEARBY FACILITIES — SAMPLE (VERIFY LOCALLY)</div>', unsafe_allow_html=True)
        for h in SAMPLE_HOSPITALS:
            maps_link = f"https://www.google.com/maps/search/{h['name'].replace(' ','+')}+{city_input.replace(' ','+')}"
            call_link = f"tel:{h['phone']}"
            st.markdown(f"""
            <div class="hosp-card">
                <div class="hosp-icon">{h['icon']}</div>
                <div style="flex:1;">
                    <div class="hosp-name">{h['name']}</div>
                    <div><span class="hosp-spec">{h['spec']}</span></div>
                    <div class="hosp-dist">📍 ~{h['dist']} &nbsp;·&nbsp; ☎ Emergency: {h['phone']}</div>
                    <div class="hosp-addr">📌 {h['addr']}, {city_input}</div>
                    <div style="display:flex;gap:1rem;margin-top:.4rem;">
                        <a href="{maps_link}" target="_blank" class="hosp-link">🗺 View on Maps →</a>
                        <a href="{call_link}" class="hosp-link">📞 Call Emergency →</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <span>ℹ️</span>
            <span>Hospital cards above are <strong>illustrative samples only</strong>.
            Distances are approximate. Always verify contact details locally or use Google Maps for accurate real-time listings.
            In a medical emergency call <strong>112</strong> immediately.</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0e1627;border:1.5px dashed #1e2d50;border-radius:12px;
            padding:2.5rem;text-align:center;color:#2a3f6f;">
            <div style="font-size:2.5rem;margin-bottom:.6rem;">📍</div>
            <div style="font-size:.9rem;color:#3d5080;font-weight:600;">Enter your city above</div>
            <div style="font-size:.82rem;margin-top:.3rem;">to find nearby hospitals, specialists and emergency services</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#2a3f6f;">
        <div style="font-size:5rem;margin-bottom:1.2rem;">🫁</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
            color:#3d5080;margin-bottom:.6rem;">No image uploaded yet</div>
        <div style="font-size:.88rem;color:#2a3a58;max-width:480px;margin:0 auto;line-height:1.8;">
            Upload a chest X-ray above to receive<br>
            <strong style="color:#3d5080;">AI diagnosis</strong> &nbsp;·&nbsp;
            <strong style="color:#3d5080;">Care guidance</strong> &nbsp;·&nbsp;
            <strong style="color:#3d5080;">Diet tips</strong><br>
            <strong style="color:#3d5080;">Hospital finder</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PNEUMOSCAN AI &nbsp;·&nbsp; PYTORCH + STREAMLIT &nbsp;·&nbsp; RESEARCH USE ONLY<br>
    <span style="color:#1e2d50;">──────────────────────────────────────────────────────</span><br>
    Not a certified medical device &nbsp;·&nbsp; Always seek professional clinical evaluation &nbsp;·&nbsp; © 2025
</div>
""", unsafe_allow_html=True)