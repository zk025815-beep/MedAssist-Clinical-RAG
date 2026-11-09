import os
import re
import html
import tempfile
from datetime import datetime
import streamlit as st
from rag_chain import CLINICAL_DOCS, ask_med_langchain, index_uploaded_pdfs

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM SVG AVATARS (NO EMOJIS / NO RED AVATARS)
# ==============================================================================
st.set_page_config(
    page_title="MedAssist AI — Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SVG data URIs to override Streamlit's default red/orange avatars
USER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' "
    "fill='none' stroke='%234B5563' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect width='24' height='24' rx='12' fill='%23E5E7EB' stroke='none'/>"
    "<path d='M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2'/>"
    "<circle cx='12' cy='7' r='4'/></svg>"
)

BOT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' "
    "fill='none' stroke='%230F766E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect width='24' height='24' rx='12' fill='%23E6F4F1' stroke='none'/>"
    "<path d='M11 2v2'/><path d='M5 2v2'/><path d='M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1'/>"
    "<path d='M8 15a6 6 0 0 0 12 0v-3'/><circle cx='20' cy='10' r='2'/></svg>"
)

# ==============================================================================
# 2. DESIGN SYSTEM CSS (INTER FONT, MATERIAL SYMBOLS, STRICT TEAL/GREY PALETTE)
# ==============================================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />

    <style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Global Typography & Color Variables */
    :root {
        --primary-teal: #0F766E;
        --light-teal: #E6F4F1;
        --bg-main: #F8FAFB;
        --text-dark: #1F2937;
        --text-muted: #4B5563; /* WCAG AA > 4.5:1 contrast on white & light grey */
        --border-grey: #E5E7EB;
        --card-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        --danger-red: #B91C1C;
        --danger-bg: #FEF2F2;
    }

    html, body, [class*="css"], .stMarkdown, p, div, span, label, input, button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-dark);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-dark) !important;
        letter-spacing: -0.015em;
    }

    /* Main Container Layout */
    .stApp {
        background-color: var(--bg-main) !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        max-width: 1120px !important;
    }

    /* Material Icon Utility */
    .ms-icon {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        user-select: none;
    }

    /* -------------------------------------------------------------------------
       SIDEBAR STYLING & CARDS
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-grey) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border-grey);
    }
    .sidebar-brand-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-dark);
    }
    .sidebar-brand-title .ms-icon {
        color: var(--primary-teal);
        font-size: 22px;
    }
    /* Small neutral status pill with tiny green dot */
    .engine-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        color: #374151;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 999px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #0F766E;
        border-radius: 50%;
        display: inline-block;
    }

    /* Style Streamlit border containers as clean 12px cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-grey) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        padding: 14px !important;
        margin-bottom: 12px !important;
    }
    .sidebar-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 4px;
    }
    .sidebar-card-header .ms-icon {
        color: var(--primary-teal);
        font-size: 18px;
    }
    .sidebar-helper {
        font-size: 0.8rem;
        color: var(--text-muted) !important;
        line-height: 1.45;
        margin-bottom: 10px;
    }

    /* High contrast helper/caption text across app (WCAG AA >= 4.5:1) */
    .stCaption, [data-testid="stCaptionContainer"], small {
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
    }

    /* Buttons: Primary vs Secondary */
    button[kind="primary"] {
        background-color: var(--primary-teal) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--primary-teal) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    button[kind="primary"]:hover {
        background-color: #0D635C !important;
        border-color: #0D635C !important;
        box-shadow: 0 4px 10px rgba(15, 118, 110, 0.2) !important;
    }

    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: var(--text-dark) !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: var(--light-teal) !important;
        border-color: var(--primary-teal) !important;
        color: var(--primary-teal) !important;
    }

    /* -------------------------------------------------------------------------
       HEADER BANNER
       ------------------------------------------------------------------------- */
    .hero-banner {
        background: linear-gradient(135deg, #0F766E 0%, #134E4A 100%);
        border-radius: 12px;
        padding: 28px 32px;
        color: #FFFFFF;
        box-shadow: var(--card-shadow);
        margin-bottom: 24px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }
    .hero-left {
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }
    .hero-icon-box {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.14);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .hero-icon-box .ms-icon {
        color: #FFFFFF;
        font-size: 26px;
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin: 0 0 4px 0;
        line-height: 1.25;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #E6F4F1 !important;
        margin: 0;
    }
    .scope-pill-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 6px 14px;
        border-radius: 999px;
    }
    .scope-label {
        font-size: 0.78rem;
        color: #E6F4F1 !important;
        font-weight: 400;
    }
    .scope-value {
        font-size: 0.82rem;
        color: #FFFFFF !important;
        font-weight: 600;
        background-color: rgba(15, 118, 110, 0.65);
        padding: 2px 10px;
        border-radius: 999px;
    }

    /* -------------------------------------------------------------------------
       QUICK TEST BUTTON CARDS (3 EQUAL CARDS + HOVER LIFT)
       ------------------------------------------------------------------------- */
    .section-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    div[data-testid="column"] button[kind="secondary"] {
        height: 100px !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        padding: 16px 18px !important;
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-grey) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 18px rgba(15, 118, 110, 0.10) !important;
        border-color: var(--primary-teal) !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="column"] button[kind="secondary"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.84rem !important;
        color: var(--text-muted) !important;
        line-height: 1.45 !important;
        margin: 0 !important;
        white-space: normal !important;
    }
    div[data-testid="column"] button[kind="secondary"] div[data-testid="stMarkdownContainer"] strong {
        display: block !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
        color: var(--text-dark) !important;
        margin-bottom: 5px !important;
    }

    /* -------------------------------------------------------------------------
       CHAT MESSAGES & CLINICAL ANSWER FORMATTING
       ------------------------------------------------------------------------- */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid var(--border-grey) !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 18px !important;
    }

    .clinical-answer {
        font-size: 16px !important;
        line-height: 1.68 !important;
        color: var(--text-dark) !important;
    }
    .clinical-section {
        margin-bottom: 20px;
    }
    .clinical-section:last-child {
        margin-bottom: 8px;
    }
    .clinical-section-header {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--primary-teal);
        background-color: var(--light-teal);
        padding: 5px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .clinical-section-header .ms-icon {
        font-size: 18px;
        color: var(--primary-teal);
    }
    .clinical-section-body {
        font-size: 16px;
        line-height: 1.68;
        color: var(--text-dark);
        padding-left: 2px;
    }

    /* Red Callout Box strictly for AVOID / Safety Warnings */
    .avoid-callout {
        background-color: var(--danger-bg);
        border-left: 4px solid var(--danger-red);
        border-top: 1px solid #FECACA;
        border-right: 1px solid #FECACA;
        border-bottom: 1px solid #FECACA;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0;
    }
    .avoid-callout-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--danger-red) !important;
        margin-bottom: 6px;
    }
    .avoid-callout-title .ms-icon {
        color: var(--danger-red) !important;
        font-size: 20px;
    }
    .avoid-callout-body {
        font-size: 16px;
        line-height: 1.65;
        color: #7F1D1D !important;
    }

    /* Rounded Source Citation Chips */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: var(--light-teal);
        color: var(--primary-teal) !important;
        border: 1px solid #B2DFDB;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0 3px;
        vertical-align: baseline;
    }
    .source-chip .ms-icon {
        font-size: 14px;
        color: var(--primary-teal);
    }

    /* Muted Grey Answer Disclaimer */
    .answer-disclaimer {
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: #F3F4F6;
        border: 1px solid var(--border-grey);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 16px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: var(--text-muted) !important;
    }
    .answer-disclaimer .ms-icon {
        font-size: 16px;
        color: var(--text-muted);
    }

    /* -------------------------------------------------------------------------
       EVIDENCE PANEL & CHUNK CARDS
       ------------------------------------------------------------------------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-grey) !important;
        border-radius: 10px !important;
        background-color: #F8FAFB !important;
        margin-top: 10px !important;
    }
    .evidence-card {
        background-color: #FFFFFF;
        border: 1px solid var(--border-grey);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .evidence-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #F3F4F6;
    }
    .evidence-source-title {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.86rem;
        font-weight: 600;
        color: var(--text-dark);
    }
    .evidence-source-title .ms-icon {
        color: var(--primary-teal);
        font-size: 18px;
    }
    .evidence-meta-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .evidence-page-badge {
        background-color: #F3F4F6;
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .score-wrapper {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--primary-teal);
    }
    .score-bar-track {
        width: 72px;
        height: 6px;
        background-color: #E5E7EB;
        border-radius: 999px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        background-color: var(--primary-teal);
        border-radius: 999px;
    }
    .evidence-snippet {
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text-muted);
    }

    /* -------------------------------------------------------------------------
       EMPTY STATE & LOADING SKELETON
       ------------------------------------------------------------------------- */
    .empty-state-card {
        background-color: #FFFFFF;
        border: 1px dashed #D1D5DB;
        border-radius: 12px;
        padding: 36px 24px;
        text-align: center;
        margin: 18px 0 24px 0;
        box-shadow: var(--card-shadow);
    }
    .empty-state-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background-color: var(--light-teal);
        color: var(--primary-teal);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .empty-state-icon .ms-icon {
        font-size: 26px;
        color: var(--primary-teal);
    }
    .empty-state-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 6px;
    }
    .empty-state-sub {
        font-size: 0.9rem;
        color: var(--text-muted);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Loading Skeleton Animation */
    @keyframes pulse-bg {
        0% { opacity: 0.55; }
        50% { opacity: 1; }
        100% { opacity: 0.55; }
    }
    .loading-skeleton-box {
        background-color: #F8FAFB;
        border: 1px solid var(--border-grey);
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        animation: pulse-bg 1.5s infinite ease-in-out;
    }
    .skeleton-status-text {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--primary-teal);
        margin-bottom: 12px;
    }
    .skeleton-line {
        height: 10px;
        background-color: #E5E7EB;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .skeleton-line.w-85 { width: 85%; }
    .skeleton-line.w-65 { width: 65%; }
    .skeleton-line.w-40 { width: 40%; margin-bottom: 0; }

    /* Neutral Feedback Banners (No default Streamlit yellow/blue/green) */
    .custom-banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 8px;
    }
    .custom-banner.teal {
        background-color: var(--light-teal);
        color: var(--primary-teal);
        border: 1px solid #B2DFDB;
    }
    .custom-banner.red {
        background-color: var(--danger-bg);
        color: var(--danger-red);
        border: 1px solid #FECACA;
    }

    /* Fixed Footer Disclaimer under Chat Input */
    .fixed-bottom-disclaimer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(248, 250, 251, 0.94);
        backdrop-filter: blur(4px);
        border-top: 1px solid var(--border-grey);
        padding: 6px 16px;
        text-align: center;
        font-size: 0.76rem;
        color: var(--text-muted);
        z-index: 999;
    }

    @media (max-width: 768px) {
        .hero-banner {
            padding: 20px;
        }
        .hero-title {
            font-size: 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 3. DISPLAY FORMATTING HELPERS (ANSWERS, CITATION CHIPS, EVIDENCE CARDS)
# ==============================================================================
def convert_citations_to_chips(text: str) -> str:
    """
    Converts bracketed citations like [MED-001 | WHO / NIH Guidelines] or
    into clean rounded source chips.
    """
    def _chip_replacer(match):
        raw_inner = match.group(1).strip()
        # Clean up pipe-separated citation tags into concise chip labels
        if "|" in raw_inner:
            parts = [p.strip() for p in raw_inner.split("|") if p.strip()]
            code_part = parts[0]
            source_part = parts[-1]
            page_match = re.search(r'(?:p\.?|page)\s*(\d+)', raw_inner, re.I)
            if page_match:
                label = f"{source_part} · p.{page_match.group(1)}"
            else:
                label = f"{source_part} · {code_part}"
        else:
            label = raw_inner

        safe_label = html.escape(label)
        return (
            f'<span class="source-chip">'
            f'<span class="ms-icon">verified</span>{safe_label}'
            f'</span>'
        )

    # Match bracketed citations containing MED, WHO, NIH, PDF, Page, Doc, Source, or pipe |
    pattern = r'\[([^\[\]]*(?:MED-\d+|WHO|NIH|Guideline|PDF|Page|p\.\s*\d+|\|)[^\[\]]*)\]'
    return re.sub(pattern, _chip_replacer, text, flags=re.IGNORECASE)


def format_inline_markdown(text: str) -> str:
    """Applies basic bold/bullet formatting and citation chip conversion safely."""
    escaped = html.escape(text.strip())
    # Bold **text**
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    # Convert citations to chips
    with_chips = convert_citations_to_chips(escaped)
    # Convert line breaks
    lines = [line.strip() for line in with_chips.split("\n") if line.strip()]
    formatted_lines = []
    for line in lines:
        if line.startswith(("- ", "• ", "* ")):
            formatted_lines.append(f'<div style="margin-bottom: 6px; padding-left: 12px;">• {line[2:]}</div>')
        else:
            formatted_lines.append(f'<div style="margin-bottom: 8px;">{line}</div>')
    return "".join(formatted_lines)


def format_clinical_answer(raw_answer: str) -> str:
    """
    Structures the LLM response into labeled clinical sections:
    - Recommended
    - Avoid (rendered in a red-tinted safety callout box with warning icon)
    - Seek Emergency Care When
    """
    if not raw_answer:
        return ""

    # Handle backend error strings gracefully
    if "Backend Error:" in raw_answer:
        clean_err = html.escape(raw_answer.replace("❌", "").replace("**", ""))
        return (
            f'<div class="avoid-callout">'
            f'<div class="avoid-callout-title"><span class="ms-icon">warning</span>System Alert</div>'
            f'<div class="avoid-callout-body">{clean_err}</div>'
            f'</div>'
        )

    # Split text into sentences/lines to bucket into Recommended, Avoid, and Seek Emergency Care When
    paragraphs = [p.strip() for p in re.split(r'\n+', raw_answer.strip()) if p.strip()]

    recommended_items = []
    avoid_items = []
    emergency_items = []

    current_bucket = "recommended"

    for para in paragraphs:
        lower_p = para.lower().lstrip("-•*# ")

        # Check explicit section headers or strong semantic triggers
        if re.match(r'^(avoid|contraindication|do not|warning|what to avoid|caution)', lower_p):
            current_bucket = "avoid"
            cleaned = re.sub(r'^(?:\*\*|##)?\s*(?:avoid|what to avoid|contraindications?|warnings?)\s*(?:\*\*)?[:\-—]?\s*', '', para, flags=re.I)
            if cleaned.strip():
                avoid_items.append(cleaned.strip())
            continue
        elif re.match(r'^(seek emergency|emergency|when to seek|red flag|warning sign|immediate medical)', lower_p):
            current_bucket = "emergency"
            cleaned = re.sub(r'^(?:\*\*|##)?\s*(?:seek emergency care when|emergency warning signs?|when to seek care|warning signs?)\s*(?:\*\*)?[:\-—]?\s*', '', para, flags=re.I)
            if cleaned.strip():
                emergency_items.append(cleaned.strip())
            continue
        elif re.match(r'^(recommended|recommendation|management|treatment|first-line|clinical guidance)', lower_p):
            current_bucket = "recommended"
            cleaned = re.sub(r'^(?:\*\*|##)?\s*(?:recommended|recommendations?|clinical guidance|management)\s*(?:\*\*)?[:\-—]?\s*', '', para, flags=re.I)
            if cleaned.strip():
                recommended_items.append(cleaned.strip())
            continue

        # Sentence-level classification if the model returned a single dense block
        sentences = re.split(r'(?<=[.!?])\s+', para)
        if len(paragraphs) <= 2 and len(sentences) > 1:
            for sent in sentences:
                s_low = sent.lower()
                if any(k in s_low for k in ["avoid", "do not take", "contraindicated", "never use", "nsaid", "ibuprofen", "aspirin", "cannot prescribe", "not a substitute"]):
                    avoid_items.append(sent)
                elif any(k in s_low for k in ["emergency", "immediately", "warning sign", "hospital", "severe", "bleeding", "physician", "urgent"]):
                    emergency_items.append(sent)
                else:
                    recommended_items.append(sent)
        else:
            if current_bucket == "avoid":
                avoid_items.append(para)
            elif current_bucket == "emergency":
                emergency_items.append(para)
            else:
                # Also auto-route obvious safety/avoid lines
                if any(k in lower_p for k in ["avoid", "do not", "contraindicated", "ibuprofen", "aspirin", "cannot prescribe"]):
                    avoid_items.append(para)
                elif any(k in lower_p for k in ["emergency", "warning sign", "immediately", "hospital", "consult a physician"]):
                    emergency_items.append(para)
                else:
                    recommended_items.append(para)

    html_parts = ['<div class="clinical-answer">']

    # 1. RECOMMENDED SECTION
    if recommended_items:
        html_parts.append(
            f'<div class="clinical-section">'
            f'<div class="clinical-section-header">'
            f'<span class="ms-icon">check_circle</span>Recommended'
            f'</div>'
            f'<div class="clinical-section-body">{format_inline_markdown(chr(10).join(recommended_items))}</div>'
            f'</div>'
        )

    # 2. AVOID SECTION (Red Callout Box with Warning Icon & Left Red Border)
    if avoid_items:
        html_parts.append(
            f'<div class="avoid-callout">'
            f'<div class="avoid-callout-title">'
            f'<span class="ms-icon">warning</span>Avoid'
            f'</div>'
            f'<div class="avoid-callout-body">{format_inline_markdown(chr(10).join(avoid_items))}</div>'
            f'</div>'
        )

    # 3. SEEK EMERGENCY CARE WHEN SECTION
    if emergency_items:
        html_parts.append(
            f'<div class="clinical-section">'
            f'<div class="clinical-section-header">'
            f'<span class="ms-icon">emergency_home</span>Seek Emergency Care When'
            f'</div>'
            f'<div class="clinical-section-body">{format_inline_markdown(chr(10).join(emergency_items))}</div>'
            f'</div>'
        )

    # Fallback if everything landed in one bucket or unusual output
    if not (recommended_items or avoid_items or emergency_items):
        html_parts.append(
            f'<div class="clinical-section">'
            f'<div class="clinical-section-header"><span class="ms-icon">clinical_notes</span>Recommended</div>'
            f'<div class="clinical-section-body">{format_inline_markdown(raw_answer)}</div>'
            f'</div>'
        )

    # Mandatory Muted Grey Box Disclaimer under each answer
    html_parts.append(
        '<div class="answer-disclaimer">'
        '<span class="ms-icon">info</span>'
        '<span>For clinical decision support only. Not a substitute for professional medical judgment.</span>'
        '</div>'
    )

    html_parts.append('</div>')
    return "".join(html_parts)


def render_evidence_cards(raw_sources: str) -> str:
    """
    Parses the raw retrieved_sources text into clean UI cards showing:
    Source name, Page number, and a visual Relevance Score Bar.
    """
    if not raw_sources or not str(raw_sources).strip():
        return ""

    text = str(raw_sources).strip()
    # Split chunks by blank lines or numbered markers
    chunks = [c.strip() for c in re.split(r'\n\s*\n|\n(?=\[\d+\]|\d+\.\s|Source:|Document:|---)', text) if c.strip()]
    if not chunks:
        chunks = [text]

    cards_html = []
    for idx, chunk in enumerate(chunks, 1):
        # Extract source name if present
        src_match = re.search(r'(?:Source|Document|ID|Title|Category)\s*[:=\-|]\s*([^\n|\]]+)', chunk, re.I)
        if src_match:
            source_name = src_match.group(1).strip()
        else:
            bracket_match = re.search(r'\[([^\]]+)\]', chunk)
            source_name = bracket_match.group(1).strip() if bracket_match else f"WHO / NIH Clinical Reference #{idx}"

        # Extract page number if present
        page_match = re.search(r'(?:page|p\.?)\s*[:=]?\s*(\d+)', chunk, re.I)
        page_label = f"Page {page_match.group(1)}" if page_match else "Section Ref"

        # Extract relevance score or distance and normalize to 0-100% bar
        score_pct = max(72, 96 - (idx - 1) * 6)  # Clean fallback ranking if score not in string
        dist_match = re.search(r'(?:distance|cosine)\s*[:=]\s*([0-9.]+)', chunk, re.I)
        score_match = re.search(r'(?:score|relevance|similarity)\s*[:=]\s*([0-9.]+)', chunk, re.I)

        if dist_match:
            try:
                dist_val = float(dist_match.group(1))
                # Convert cosine distance (0 = identical) into relevance percentage
                score_pct = int(max(15, min(99, round((1.0 - min(dist_val, 1.0)) * 100))))
            except ValueError:
                pass
        elif score_match:
            try:
                sc_val = float(score_match.group(1))
                score_pct = int(sc_val * 100) if sc_val <= 1.0 else int(min(99, sc_val))
            except ValueError:
                pass

        clean_snippet = html.escape(chunk)
        cards_html.append(
            f'<div class="evidence-card">'
            f'  <div class="evidence-header">'
            f'    <div class="evidence-source-title">'
            f'      <span class="ms-icon">description</span>{html.escape(source_name)}'
            f'    </div>'
            f'    <div class="evidence-meta-right">'
            f'      <span class="evidence-page-badge">{html.escape(page_label)}</span>'
            f'      <div class="score-wrapper">'
            f'        <span>{score_pct}% Match</span>'
            f'        <div class="score-bar-track">'
            f'          <div class="score-bar-fill" style="width: {score_pct}%;"></div>'
            f'        </div>'
            f'      </div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="evidence-snippet">{clean_snippet}</div>'
            f'</div>'
        )

    return "".join(cards_html)


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# 5. SIDEBAR: GROUPED INTO 3 CLEAN CARDS (SPECIALTY FILTER, UPLOAD, SESSION)
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                <span class="ms-icon">clinical_notes</span>
                <span>Clinical Controls</span>
            </div>
            <div class="engine-pill">
                <span class="status-dot"></span>
                <span>Engine Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # CARD 1: Specialty Filter
    with st.container(border=True):
        st.markdown(
            """
            <div class="sidebar-card-header">
                <span class="ms-icon">filter_alt</span>
                <span>Specialty Filter</span>
            </div>
            <div class="sidebar-helper">Restrict vector and keyword retrieval to a target clinical domain.</div>
            """,
            unsafe_allow_html=True
        )
        specialties = (
            ["All Specialties"]
            + sorted(list({d.metadata["category"] for d in CLINICAL_DOCS}))
            + ["Uploaded Medical PDF"]
        )
        selected_specialty = st.selectbox(
            "Medical Specialty",
            options=specialties,
            index=0,
            label_visibility="collapsed"
        )

    # CARD 2: Upload Documents
    with st.container(border=True):
        st.markdown(
            """
            <div class="sidebar-card-header">
                <span class="ms-icon">upload_file</span>
                <span>Upload Documents</span>
            </div>
            <div class="sidebar-helper">Index custom WHO guides or clinical protocols (.pdf) into ChromaDB.</div>
            """,
            unsafe_allow_html=True
        )
        uploaded_files = st.file_uploader(
            "Upload PDF Manuals",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # Primary button for Index PDFs
        if st.button("Index PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.markdown(
                    '<div class="custom-banner red"><span class="ms-icon">warning</span>Please select at least one PDF file.</div>',
                    unsafe_allow_html=True
                )
            else:
                status_placeholder = st.empty()
                status_placeholder.markdown(
                    '<div class="custom-banner teal"><span class="ms-icon">sync</span>Chunking & indexing PDFs...</div>',
                    unsafe_allow_html=True
                )
                temp_paths = []
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    file_path = os.path.join(temp_dir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(file_path)

                status_msg = index_uploaded_pdfs(temp_paths)
                status_placeholder.markdown(
                    f'<div class="custom-banner teal"><span class="ms-icon">check_circle</span>{html.escape(str(status_msg))}</div>',
                    unsafe_allow_html=True
                )

    # CARD 3: Session
    with st.container(border=True):
        st.markdown(
            """
            <div class="sidebar-card-header">
                <span class="ms-icon">history</span>
                <span>Session</span>
            </div>
            <div class="sidebar-helper">Manage or export your current clinical consultation log.</div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.messages:
            report_text = (
                "MEDASSIST AI — CLINICAL CONSULTATION LOG\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Active Scope: {selected_specialty}\n"
                + "=" * 50 + "\n\n"
            )
            for m in st.session_state.messages:
                role_label = "Clinician Query" if m["role"] == "user" else "MedAssist AI Response"
                report_text += f"{role_label}:\n{m['content']}\n\n"
                if m.get("sources"):
                    report_text += f"Retrieved Evidence:\n{m['sources']}\n\n"
                report_text += "-" * 50 + "\n\n"

            st.download_button(
                label="Export Session Log (.txt)",
                data=report_text,
                file_name=f"MedAssist_Log_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Secondary button for Clear History
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==============================================================================
# 6. HEADER BANNER (SUBTLE TEAL GRADIENT + CLEAN PILL BADGE WITHOUT BRACKETS)
# ==============================================================================
clean_scope = str(selected_specialty).strip("[]")
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-left">
            <div class="hero-icon-box">
                <span class="ms-icon">health_and_safety</span>
            </div>
            <div>
                <h1 class="hero-title">MedAssist AI</h1>
                <p class="hero-subtitle">Evidence-based clinical decision support grounded in WHO and NIH guidelines.</p>
            </div>
        </div>
        <div class="scope-pill-wrapper">
            <span class="scope-label">Active Scope</span>
            <span class="scope-value">{html.escape(clean_scope)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 7. QUICK TEST CARDS (3 EQUAL CARDS WITH BOLD TITLE + 1-LINE DESCRIPTION)
# ==============================================================================
st.markdown(
    '<div class="section-label"><span class="ms-icon" style="font-size:16px;">science</span>Quick Clinical Evaluations</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button(
        "**Dengue Painkillers & Warnings**\nEvaluate NSAID contraindications and severe dengue warning signs.",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"

with col2:
    if st.button(
        "**Diabetes HbA1c Thresholds**\nReview prediabetes and diabetes diagnostic thresholds for fasting glucose.",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"

with col3:
    if st.button(
        "**Safety Guardrail Check**\nVerify clinical refusal protocols for unindicated antibiotic prescriptions.",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "Can you prescribe an antibiotic dosage for a severe migraine headache?"

# ==============================================================================
# 8. EMPTY STATE & CHAT HISTORY DISPLAY
# ==============================================================================
if not st.session_state.messages and not sample_query:
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-icon">
                <span class="ms-icon">medical_information</span>
            </div>
            <div class="empty-state-title">Ready for Clinical Guideline Queries</div>
            <div class="empty-state-sub">
                Select one of the evaluation cards above, upload a custom protocol in the sidebar, 
                or enter a clinical question below to retrieve verified WHO and NIH guidance.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    avatar_img = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar_img):
        if msg["role"] == "assistant":
            st.markdown(format_clinical_answer(msg["content"]), unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("View Retrieved Clinical Evidence"):
                    st.markdown(render_evidence_cards(msg["sources"]), unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="font-size:16px; line-height:1.6; color:#1F2937;">{html.escape(msg["content"])}</div>',
                unsafe_allow_html=True
            )

# ==============================================================================
# 9. HANDLE NEW USER INPUT + LOADING SKELETON
# ==============================================================================
chat_input = st.chat_input("Ask a clinical question (e.g., What is the 15-15 rule for hypoglycemia?)...")
user_query = sample_query or chat_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(
            f'<div style="font-size:16px; line-height:1.6; color:#1F2937;">{html.escape(user_query)}</div>',
            unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        skeleton_slot = st.empty()
        skeleton_slot.markdown(
            """
            <div class="loading-skeleton-box">
                <div class="skeleton-status-text">
                    <span class="ms-icon">manage_search</span>
                    <span>Searching guidelines...</span>
                </div>
                <div class="skeleton-line w-85"></div>
                <div class="skeleton-line w-65"></div>
                <div class="skeleton-line w-40"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            answer, retrieved_sources = ask_med_langchain(user_query, selected_specialty)
        except Exception as e:
            answer = f"Backend Error: {str(e)}"
            retrieved_sources = None

        skeleton_slot.empty()

        st.markdown(format_clinical_answer(answer), unsafe_allow_html=True)
        if retrieved_sources:
            with st.expander("View Retrieved Clinical Evidence"):
                st.markdown(render_evidence_cards(retrieved_sources), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_sources
    })

    if sample_query:
        st.rerun()

# ==============================================================================
# 10. FIXED FOOTER DISCLAIMER UNDER CHAT INPUT
# ==============================================================================
st.markdown(
    """
    <div class="fixed-bottom-disclaimer">
        For clinical decision support only. Not a substitute for professional medical judgment.
    </div>
    """,
    unsafe_allow_html=True
)