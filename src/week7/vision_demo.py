import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# ── Page config MUST be first ─────────────────────────────────────
st.set_page_config(
    page_title="Azure Vision AI",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Professional CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0D1F3C 0%, #080C14 50%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* Remove default padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── HERO HEADER ── */
.hero {
    background: linear-gradient(135deg, #0A1628 0%, #0D2040 50%, #0A1628 100%);
    border-bottom: 1px solid rgba(99,179,237,0.15);
    padding: 48px 64px 40px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 500;
    color: #63B3ED;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #63B3ED;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 52px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 12px !important;
}
.hero h1 span { color: #63B3ED; }
.hero p {
    font-size: 17px !important;
    color: #94A3B8 !important;
    font-weight: 300 !important;
    max-width: 520px !important;
    line-height: 1.6 !important;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 32px;
}
.hero-stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.hero-stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #63B3ED;
}
.hero-stat-label {
    font-size: 12px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── MAIN LAYOUT ── */
.main-layout {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 0;
    min-height: calc(100vh - 200px);
}

/* ── LEFT PANEL ── */
.left-panel {
    background: #0A1220;
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 40px 32px;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* ── INPUT CARD ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.input-card:hover { border-color: rgba(99,179,237,0.2); }
.input-card-title {
    font-size: 13px;
    font-weight: 500;
    color: #CBD5E1;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.input-icon {
    width: 28px; height: 28px;
    background: rgba(99,179,237,0.1);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

/* ── FEATURES TOGGLE ── */
.features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 16px;
}
.feature-chip {
    background: rgba(99,179,237,0.07);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    color: #94A3B8;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
}
.feature-chip.active {
    background: rgba(99,179,237,0.15);
    border-color: rgba(99,179,237,0.4);
    color: #63B3ED;
}

/* ── RIGHT PANEL ── */
.right-panel {
    background: #080C14;
    padding: 40px 48px;
}

/* ── RESULT CARDS ── */
.result-section {
    margin-bottom: 32px;
}
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
}
.result-icon-wrap {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #E2E8F0;
}
.result-subtitle { font-size: 12px; color: #475569; }

/* Caption card */
.caption-card {
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(99,179,237,0.03));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 28px;
    position: relative;
    overflow: hidden;
}
.caption-card::before {
    content: '"';
    position: absolute;
    top: -10px; left: 16px;
    font-size: 80px;
    color: rgba(99,179,237,0.1);
    font-family: Georgia, serif;
    line-height: 1;
}
.caption-text {
    font-size: 20px;
    font-weight: 300;
    color: #E2E8F0;
    font-style: italic;
    line-height: 1.5;
    margin-bottom: 12px;
}
.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 12px;
    color: #63B3ED;
    font-weight: 500;
}

/* Tags */
.tags-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 13px;
    color: #CBD5E1;
    transition: all 0.2s;
}
.tag-pill:hover {
    background: rgba(99,179,237,0.1);
    border-color: rgba(99,179,237,0.25);
    color: #63B3ED;
}
.tag-conf {
    font-size: 11px;
    color: #475569;
    font-weight: 500;
}
.tag-bar {
    width: 32px; height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
}
.tag-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3B82F6, #63B3ED);
    border-radius: 2px;
}

/* Objects */
.objects-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
}
.object-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s;
}
.object-card:hover {
    background: rgba(99,179,237,0.05);
    border-color: rgba(99,179,237,0.2);
    transform: translateY(-2px);
}
.object-emoji { font-size: 28px; margin-bottom: 8px; }
.object-name {
    font-size: 13px;
    font-weight: 500;
    color: #CBD5E1;
    text-transform: capitalize;
    margin-bottom: 4px;
}
.object-conf { font-size: 11px; color: #475569; }

/* People */
.people-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(139,92,246,0.03));
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.people-count {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    color: #A78BFA;
    line-height: 1;
}
.people-label { font-size: 14px; color: #94A3B8; }

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    text-align: center;
    gap: 16px;
}
.empty-icon {
    font-size: 64px;
    opacity: 0.2;
}
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #334155;
}
.empty-sub { font-size: 14px; color: #1E293B; }

/* Divider */
.divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 32px 0;
}

/* Streamlit overrides */
.stRadio > div { gap: 0 !important; }
.stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    margin-bottom: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: #94A3B8 !important;
    font-size: 14px !important;
}
.stRadio label:hover {
    border-color: rgba(99,179,237,0.3) !important;
    color: #E2E8F0 !important;
}
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}
.stFileUploader {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(99,179,237,0.2) !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1E40AF, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.4) !important;
}
.stSpinner > div { border-top-color: #63B3ED !important; }
.stImage { border-radius: 12px; overflow: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="hero-dot"></div>
        Week 7 · Azure AI Services
    </div>
    <h1>Computer <span>Vision</span><br>Intelligence</h1>
    <p>Powered by Azure AI Vision 4.0 — Florence model trained on 900M image-text pairs</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-num">4.0</span>
            <span class="hero-stat-label">API Version</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-num">900M</span>
            <span class="hero-stat-label">Training Images</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-num">&lt;300ms</span>
            <span class="hero-stat-label">Response Time</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-num">0</span>
            <span class="hero-stat-label">Training Needed</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────────
left, right = st.columns([4, 7], gap="small")

with left:
    st.markdown("""
    <div style="background:#0A1220; padding:32px; border-right:1px solid rgba(255,255,255,0.06); min-height:100vh;">
    <div class="panel-title">Input Configuration</div>
    """, unsafe_allow_html=True)

    # Input mode
    st.markdown('<div class="input-card"><div class="input-card-title"><div class="input-icon">📥</div>Input Method</div>', unsafe_allow_html=True)
    mode = st.radio("", ["🔗  Image URL", "📁  Upload File"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    image_bytes = None
    url = None

    if "URL" in mode:
        st.markdown('<div class="input-card"><div class="input-card-title"><div class="input-icon">🌐</div>Image URL</div>', unsafe_allow_html=True)
        url = st.text_input("", placeholder="https://images.unsplash.com/...", label_visibility="collapsed")
        if url:
            try:
                st.image(url, use_container_width=True)
            except:
                st.error("Cannot preview this URL")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="input-card"><div class="input-card-title"><div class="input-icon">📁</div>Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if uploaded:
            image_bytes = uploaded.read()
            st.image(image_bytes, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Sample URLs
    st.markdown("""
    <div style="margin-top:8px; margin-bottom:16px;">
    <div class="panel-title" style="margin-bottom:12px;">Quick Test URLs</div>
    </div>
    """, unsafe_allow_html=True)

    samples = {
        "🏙️ City": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800",
        "🐕 Dog": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",
        "🍕 Food": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800",
        "🌿 Nature": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
    }

    cols = st.columns(2)
    for i, (label, sample_url) in enumerate(samples.items()):
        with cols[i % 2]:
            if st.button(label, key=f"sample_{i}"):
                st.session_state["sample_url"] = sample_url
                st.rerun()

    if "sample_url" in st.session_state and "URL" in mode:
        url = st.session_state["sample_url"]
        st.image(url, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("⚡  Analyze Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT PANEL — RESULTS ─────────────────────────────────────────
with right:
    st.markdown('<div style="padding:32px 40px; background:#080C14; min-height:100vh;">', unsafe_allow_html=True)

    if not analyze:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">👁️</div>
            <div class="empty-title">Ready to Analyze</div>
            <div class="empty-sub">Choose an image source and click Analyze Image</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Validate input
        has_input = (url and "URL" in mode) or (image_bytes and "Upload" in mode)
        if not has_input:
            st.error("Please provide an image first.")
        else:
            try:
                from azure.ai.vision.imageanalysis import ImageAnalysisClient
                from azure.ai.vision.imageanalysis.models import VisualFeatures
                from azure.core.credentials import AzureKeyCredential

                ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")
                KEY = os.getenv("AZURE_VISION_KEY") or os.getenv("AZURE_COMPUTER_VISION_KEY")

                client = ImageAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

                with st.spinner("Analyzing with Azure Vision AI..."):
                    features = [VisualFeatures.CAPTION, VisualFeatures.TAGS, VisualFeatures.OBJECTS, VisualFeatures.PEOPLE]

                    if "URL" in mode and url:
                        result = client.analyze_from_url(image_url=url, visual_features=features)
                    else:
                        result = client.analyze(image_data=image_bytes, visual_features=features)

                # ── CAPTION ──────────────────────────────────────
                if result.caption:
                    conf = result.caption.confidence
                    conf_color = "#22C55E" if conf > 0.8 else "#F59E0B" if conf > 0.5 else "#EF4444"
                    st.markdown(f"""
                    <div class="result-section">
                        <div class="result-header">
                            <div class="result-icon-wrap" style="background:rgba(99,179,237,0.1)">💬</div>
                            <div>
                                <div class="result-title">AI Caption</div>
                                <div class="result-subtitle">Natural language description of the image</div>
                            </div>
                        </div>
                        <div class="caption-card">
                            <div class="caption-text">{result.caption.text}</div>
                            <span class="confidence-badge">
                                <span style="color:{conf_color}">●</span>
                                {conf:.1%} confidence
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # ── TAGS ─────────────────────────────────────────
                if result.tags and result.tags.list:
                    tags = [t for t in result.tags.list if t.confidence > 0.5][:12]
                    tags_html = ""
                    for t in tags:
                        bar_width = int(t.confidence * 100)
                        tags_html += f"""
                        <div class="tag-pill">
                            {t.name}
                            <div class="tag-bar"><div class="tag-bar-fill" style="width:{bar_width}%"></div></div>
                            <span class="tag-conf">{t.confidence:.0%}</span>
                        </div>"""

                    st.markdown(f"""
                    <div class="result-section">
                        <div class="result-header">
                            <div class="result-icon-wrap" style="background:rgba(34,197,94,0.1)">🏷️</div>
                            <div>
                                <div class="result-title">Tags Detected</div>
                                <div class="result-subtitle">{len(tags)} tags above 50% confidence threshold</div>
                            </div>
                        </div>
                        <div class="tags-wrap">{tags_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # ── OBJECTS ──────────────────────────────────────
                if result.objects and result.objects.list:
                    obj_emojis = {
                        "person":"👤","car":"🚗","building":"🏢","tree":"🌲",
                        "dog":"🐕","cat":"🐈","chair":"🪑","table":"🪑",
                        "phone":"📱","laptop":"💻","food":"🍽️","bottle":"🍾"
                    }
                    objs = result.objects.list[:8]
                    cards_html = ""
                    for obj in objs:
                        name = obj.tags[0].name if obj.tags else "object"
                        conf = obj.tags[0].confidence if obj.tags else 0
                        emoji = obj_emojis.get(name.lower(), "📦")
                        cards_html += f"""
                        <div class="object-card">
                            <div class="object-emoji">{emoji}</div>
                            <div class="object-name">{name}</div>
                            <div class="object-conf">{conf:.0%}</div>
                        </div>"""

                    st.markdown(f"""
                    <div class="result-section">
                        <div class="result-header">
                            <div class="result-icon-wrap" style="background:rgba(245,158,11,0.1)">📦</div>
                            <div>
                                <div class="result-title">Objects Detected</div>
                                <div class="result-subtitle">{len(objs)} objects with bounding box coordinates</div>
                            </div>
                        </div>
                        <div class="objects-grid">{cards_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # ── PEOPLE ───────────────────────────────────────
                if result.people and result.people.list:
                    confident_people = [p for p in result.people.list if p.confidence > 0.5]
                    count = len(confident_people)
                    st.markdown(f"""
                    <div class="result-section">
                        <div class="result-header">
                            <div class="result-icon-wrap" style="background:rgba(139,92,246,0.1)">👥</div>
                            <div>
                                <div class="result-title">People Detected</div>
                                <div class="result-subtitle">Persons with confidence above 50%</div>
                            </div>
                        </div>
                        <div class="people-card">
                            <div class="people-count">{count}</div>
                            <div>
                                <div class="people-label">{"person detected" if count == 1 else "people detected"}</div>
                                <div style="font-size:12px;color:#475569;margin-top:4px;">
                                    {len(result.people.list) - count} below threshold · filtered out
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── SUCCESS FOOTER ────────────────────────────────
                st.markdown("""
                <div style="margin-top:32px; padding:16px 20px; background:rgba(34,197,94,0.05);
                     border:1px solid rgba(34,197,94,0.15); border-radius:12px;
                     display:flex; align-items:center; gap:12px;">
                    <span style="font-size:20px">✅</span>
                    <div>
                        <div style="font-size:14px;font-weight:500;color:#86EFAC;">Analysis Complete</div>
                        <div style="font-size:12px;color:#475569;">Azure Vision API · Florence Model · 4.0</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div style="padding:24px; background:rgba(239,68,68,0.05);
                     border:1px solid rgba(239,68,68,0.2); border-radius:16px;">
                    <div style="font-size:16px;font-weight:600;color:#FCA5A5;margin-bottom:8px;">
                        ⚠️ Analysis Failed
                    </div>
                    <div style="font-size:13px;color:#94A3B8;">{str(e)}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)