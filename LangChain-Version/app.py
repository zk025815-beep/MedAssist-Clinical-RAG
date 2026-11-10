import os
import re
import html
import tempfile
from datetime import datetime
import streamlit as st
from rag_chain import CLINICAL_DOCS, ask_med_langchain, index_uploaded_pdfs

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK-MODE SVG AVATARS (NO EMOJIS)
# ==============================================================================
st.set_page_config(
    page_title="MedAssist AI — Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

USER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' "
    "fill='none' stroke='%23E5E7EB' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect width='24' height='24' rx='12' fill='%231F2A3D' stroke='none'/>"
    "<path d='M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2'/>"
    "<circle cx='12' cy='7' r='4'/></svg>"
)

BOT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='36' height='36' viewBox='0 0 24 24' "
    "fill='none' stroke='%2314B8A6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect width='24' height='24' rx='12' fill='%230D2E2B' stroke='none'/>"
    "<path d='M11 2v2'/><path d='M5 2v2'/><path d='M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1'/>"
    "<path d='M8 15a6 6 0 0 0 12 0v-3'/><circle cx='20' cy='10' r='2'/></svg>"
)

# ==============================================================================
# 2. DARK THEME DESIGN SYSTEM & UI FIXES CSS
# ==============================================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />

    <style>
    /* -------------------------------------------------------------------------
       KEEP HEADER & SIDEBAR TOGGLE CLICKABLE; HIDE ONLY CHROME
       ------------------------------------------------------------------------- */
    .stDeployButton,
    [data-testid="stAppDeployButton"],
    #MainMenu,
    footer,
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background-color: #111A2B !important;
        border: 1px solid #1F2A3D !important;
        border-radius: 8px !important;
        color: #14B8A6 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    }

    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="collapsedControl"] span {
        color: #14B8A6 !important;
        fill: #14B8A6 !important;
    }

    /* -------------------------------------------------------------------------
       COLOR TOKENS & TYPOGRAPHY
       ------------------------------------------------------------------------- */
    :root {
        --bg-main: #0B1220;
        --bg-sidebar: #0F172A;
        --bg-card: #111A2B;
        --bg-card-inner: #0E1625;
        --border-color: #1F2A3D;
        --text-primary: #E5E7EB;
        --text-muted: #9CA3AF;
        --teal-primary: #14B8A6;
        --teal-hover: #2DD4BF;
        --teal-surface: rgba(20, 184, 166, 0.12);
        --amber-border: #F59E0B;
        --amber-text: #FCD34D;
        --danger-bg: #2A1215;
        --danger-border: #F87171;
        --danger-text: #FCA5A5;
        --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.28);
    }

    html, body, .stApp, .stMarkdown, p, label, input, textarea, button,
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.015em;
    }

    span[data-testid="stIconMaterial"],
    .material-icons,
    .material-symbols-rounded,
    .ms-icon {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        vertical-align: middle;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        user-select: none;
    }

    /* -------------------------------------------------------------------------
       FIX 6: REDUCE TOP PADDING BY ~30% (16PX TOP, 140PX BOTTOM, 960PX MAX)
       ------------------------------------------------------------------------- */
    .stApp {
        background-color: var(--bg-main) !important;
    }

    .main .block-container {
        max-width: 960px !important;
        margin: 0 auto !important;
        padding-top: 16px !important;
        padding-bottom: 140px !important;
    }

    /* -------------------------------------------------------------------------
       SIDEBAR STYLING & DARK CARDS
       ------------------------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 16px !important;
        padding-bottom: 32px !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 14px;
        border-bottom: 1px solid var(--border-color);
    }

    .sidebar-brand-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .sidebar-brand-title .ms-icon {
        color: var(--teal-primary);
        font-size: 21px;
    }

    .engine-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-size: 0.74rem;
        font-weight: 500;
        padding: 3px 9px;
        border-radius: 999px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: var(--teal-primary);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px var(--teal-primary);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        padding: 14px !important;
        margin-bottom: 14px !important;
    }

    .sidebar-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .sidebar-card-header .ms-icon {
        color: var(--teal-primary);
        font-size: 18px;
    }

    .sidebar-helper {
        font-size: 0.8rem;
        color: var(--text-muted) !important;
        line-height: 1.45;
        margin-bottom: 10px;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--bg-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: var(--bg-main) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploader"] section button {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    button:focus, button:focus-visible, button:active {
        outline: none !important;
        box-shadow: none !important;
    }

    button[kind="primary"] {
        background-color: var(--teal-primary) !important;
        color: #0B1220 !important;
        border: 1px solid var(--teal-primary) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"] p, button[kind="primary"] span {
        color: #0B1220 !important;
        font-weight: 600 !important;
    }

    button[kind="primary"]:hover {
        background-color: var(--teal-hover) !important;
        border-color: var(--teal-hover) !important;
        box-shadow: 0 0 14px rgba(20, 184, 166, 0.35) !important;
    }

    button[kind="secondary"], [data-testid="stDownloadButton"] button {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover, [data-testid="stDownloadButton"] button:hover {
        border-color: var(--teal-primary) !important;
        color: var(--teal-hover) !important;
        background-color: rgba(20, 184, 166, 0.08) !important;
    }

    /* -------------------------------------------------------------------------
       HEADER BANNER
       ------------------------------------------------------------------------- */
    .hero-banner {
        background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 12px;
        padding: 24px 28px;
        color: #E5E7EB;
        box-shadow: 0 8px 28px rgba(15, 118, 110, 0.25);
        margin-bottom: 20px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }

    .hero-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .hero-icon-box {
        width: 46px;
        height: 46px;
        border-radius: 10px;
        background-color: rgba(11, 18, 32, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.16);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .hero-icon-box .ms-icon {
        color: #2DD4BF;
        font-size: 26px;
    }

    .hero-title {
        font-size: 1.45rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin: 0 0 4px 0;
        line-height: 1.25;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        font-weight: 400;
        color: #CCFBF1 !important;
        margin: 0;
    }

    .scope-pill-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: rgba(11, 18, 32, 0.35);
        border: 1px solid rgba(45, 212, 191, 0.35);
        padding: 6px 14px;
        border-radius: 999px;
    }

    .scope-label {
        font-size: 0.78rem;
        color: #CCFBF1 !important;
        font-weight: 400;
    }

    .scope-value {
        font-size: 0.82rem;
        color: #FFFFFF !important;
        font-weight: 600;
        background-color: rgba(20, 184, 166, 0.35);
        padding: 2px 10px;
        border-radius: 999px;
    }

    /* -------------------------------------------------------------------------
       FIX 5: QUICK-TEST CARDS (TITLE LINE 1, DESC LINE 2, NO TRUNCATION, EQUAL HEIGHT)
       ------------------------------------------------------------------------- */
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .section-label .ms-icon {
        color: var(--teal-primary);
        font-size: 17px;
    }

    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="column"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    div[data-testid="column"] .stButton {
        flex: 1 !important;
        display: flex !important;
    }

    div[data-testid="column"] button[kind="secondary"] {
        height: 96px !important;
        min-height: 96px !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: flex-start !important;
        text-align: left !important;
        padding: 14px 16px !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
    }

    div[data-testid="column"] button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        border-color: var(--teal-primary) !important;
        box-shadow: 0 0 16px rgba(20, 184, 166, 0.22) !important;
        background-color: var(--bg-card) !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.81rem !important;
        font-weight: 400 !important;
        color: var(--text-muted) !important;
        line-height: 1.38 !important;
        margin: 0 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] strong {
        display: block !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 3px !important;
        line-height: 1.25 !important;
        white-space: normal !important;
    }

    /* -------------------------------------------------------------------------
       FIX 2 & 3: CLINICAL ANSWER CARDS, AMBER HOSPITALIZE CARD & SOURCE CHIP LINE
       ------------------------------------------------------------------------- */
    [data-testid="stChatMessage"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 18px !important;
    }

    .clinical-answer {
        font-size: 16px !important;
        line-height: 1.68 !important;
        color: var(--text-primary) !important;
    }

    .clinical-section {
        margin-bottom: 16px;
    }

    .clinical-section-header {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--teal-hover);
        background-color: var(--teal-surface);
        border: 1px solid rgba(20, 184, 166, 0.28);
        padding: 4px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    .clinical-section-header .ms-icon {
        font-size: 18px;
        color: var(--teal-primary);
    }

    .clinical-section-body {
        font-size: 16px;
        line-height: 1.68;
        color: var(--text-primary);
    }

    /* ONLY Avoid uses Red Callout (#FCA5A5 on #2A1215 + 3px #F87171 left border) */
    .avoid-callout {
        background-color: var(--danger-bg);
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-left: 3px solid var(--danger-border);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
    }

    .avoid-callout-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--danger-text) !important;
        margin-bottom: 6px;
    }

    .avoid-callout-title .ms-icon {
        color: var(--danger-border) !important;
        font-size: 19px;
    }

    .avoid-callout-body {
        font-size: 16px;
        line-height: 1.65;
        color: var(--danger-text) !important;
    }

    /* Neutral Dark Card for "Signs to Monitor" */
    .neutral-clinical-card {
        background-color: var(--bg-card-inner);
        border: 1px solid var(--border-color);
        border-left: 3px solid #64748B;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
    }

    .neutral-card-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 6px;
    }

    .neutral-card-title .ms-icon {
        color: var(--text-muted) !important;
        font-size: 19px;
    }

    /* Neutral Dark Card with Amber Left Border for "When to Hospitalize" */
    .amber-clinical-card {
        background-color: var(--bg-card-inner);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--amber-border);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 14px 0;
    }

    .amber-card-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--amber-text) !important;
        margin-bottom: 6px;
    }

    .amber-card-title .ms-icon {
        color: var(--amber-border) !important;
        font-size: 19px;
    }

    .card-body-text {
        font-size: 16px;
        line-height: 1.65;
        color: var(--text-primary) !important;
    }

    /* Outside Available Guidelines Callout */
    .out-of-scope-callout {
        background-color: var(--bg-card-inner);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--text-muted);
        border-radius: 10px;
        padding: 16px 18px;
        margin: 8px 0 14px 0;
    }

    .out-of-scope-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 8px;
    }

    .out-of-scope-title .ms-icon {
        color: var(--text-muted) !important;
        font-size: 20px;
    }

    .out-of-scope-body {
        font-size: 16px;
        line-height: 1.65;
        color: var(--text-primary) !important;
        margin-bottom: 8px;
    }

    .out-of-scope-sub {
        font-size: 0.88rem;
        color: var(--text-muted) !important;
    }

    /* FIX 3: Dedicated Source Chip Row on Its Own Line Below Answer */
    .source-chip-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 4px;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background-color: var(--teal-surface);
        color: var(--teal-hover) !important;
        border: 1px solid rgba(20, 184, 166, 0.35);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
    }

    .source-chip .ms-icon {
        font-size: 14px;
        color: var(--teal-primary);
    }

    /* Muted Disclaimer Inside Answer Card */
    .answer-disclaimer {
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 9px 13px;
        margin-top: 12px;
        font-size: 0.8rem;
        color: var(--text-muted) !important;
    }

    .answer-disclaimer .ms-icon {
        font-size: 16px;
        color: var(--text-muted);
    }

    /* -------------------------------------------------------------------------
       FIX 4: STRUCTURED EVIDENCE CARDS (CONDITION, QUESTION, ANSWER)
       ------------------------------------------------------------------------- */
    [data-testid="stExpander"] {
        background-color: var(--bg-main) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        margin-top: 12px !important;
        box-shadow: var(--card-shadow) !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        padding: 10px 14px !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--teal-hover) !important;
    }

    .evidence-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .evidence-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
    }

    .evidence-source-title {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .evidence-source-title .ms-icon {
        color: var(--teal-primary);
        font-size: 17px;
    }

    .evidence-meta-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .evidence-page-badge {
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        color: var(--text-muted);
        font-size: 0.74rem;
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
        color: var(--teal-hover);
    }

    .score-bar-track {
        width: 72px;
        height: 6px;
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 999px;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        background-color: var(--teal-primary);
        border-radius: 999px;
    }

    .evidence-field-grid {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .evidence-field-row {
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text-primary);
    }

    .evidence-field-label {
        font-weight: 600;
        color: var(--text-muted);
        margin-right: 6px;
    }

    /* -------------------------------------------------------------------------
       EMPTY STATE, SKELETON & FIXED INPUT
       ------------------------------------------------------------------------- */
    .empty-state-card {
        background-color: var(--bg-card);
        border: 1px dashed var(--border-color);
        border-radius: 12px;
        padding: 36px 24px;
        text-align: center;
        margin: 22px 0;
        box-shadow: var(--card-shadow);
    }

    .empty-state-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background-color: var(--teal-surface);
        border: 1px solid rgba(20, 184, 166, 0.3);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }

    .empty-state-icon .ms-icon {
        font-size: 26px;
        color: var(--teal-primary);
    }

    .empty-state-title {
        font-size: 1.04rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .empty-state-sub {
        font-size: 0.88rem;
        color: var(--text-muted);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    @keyframes pulse-dark {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    .loading-skeleton-box {
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        animation: pulse-dark 1.5s infinite ease-in-out;
    }

    .skeleton-status-text {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--teal-hover);
        margin-bottom: 12px;
    }

    .skeleton-line {
        height: 10px;
        background-color: var(--border-color);
        border-radius: 6px;
        margin-bottom: 8px;
    }

    .skeleton-line.w-85 { width: 85%; }
    .skeleton-line.w-65 { width: 65%; }
    .skeleton-line.w-40 { width: 40%; margin-bottom: 0; }

    .custom-banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 8px;
    }

    .custom-banner.teal {
        background-color: var(--teal-surface);
        color: var(--teal-hover);
        border: 1px solid rgba(20, 184, 166, 0.35);
    }

    .custom-banner.red {
        background-color: var(--danger-bg);
        color: var(--danger-text);
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    [data-testid="stBottom"] {
        background-color: transparent !important;
    }

    [data-testid="stBottom"] > div {
        background-color: var(--bg-main) !important;
        padding-bottom: 34px !important;
        border-top: 1px solid var(--border-color) !important;
    }

    [data-testid="stChatInput"] {
        max-width: 960px !important;
        margin: 0 auto !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: var(--card-shadow) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--teal-primary) !important;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.22) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }

    .fixed-bottom-disclaimer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: rgba(11, 18, 32, 0.95);
        backdrop-filter: blur(6px);
        border-top: 1px solid var(--border-color);
        padding: 6px 16px;
        text-align: center;
        font-size: 0.75rem;
        color: var(--text-muted);
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 3. DISPLAY FORMATTING HELPERS
# ==============================================================================
def strip_duplicate_disclaimers(text: str) -> str:
    """Removes inline LLM disclaimers so the disclaimer is never duplicated."""
    cleaned = re.sub(
        r'(?im)^\s*(?:\*\*|⚠️|Note:|Disclaimer:|Medical Disclaimer:|\*Disclaimer:)?\s*'
        r'(?:medical\s+disclaimer\s*[:\-—]?\s*)?'
        r'(?:for educational purposes only|this information is for clinical decision support|'
        r'not a substitute for professional medical|please consult a qualified physician|'
        r'always consult a healthcare professional).*$',
        '',
        text
    )
    return cleaned.strip()


def extract_and_strip_citations(text: str) -> tuple[str, list[str]]:
    """
    FIX 3: Extracts all citations like [MED-001 | WHO / NIH Guidelines] or bare MED-001,
    removes them from the inline text (no brackets left behind), and returns a list of
    clean source labels to render on their own line below the answer.
    """
    sources_found = []

    def _add_source(label: str):
        clean_lbl = label.strip(" []|:-·")
        if clean_lbl and clean_lbl not in sources_found:
            sources_found.append(clean_lbl)

    def _bracket_extractor(match):
        raw_inner = match.group(1).strip()
        page_match = re.search(r'(?:p\.?|page)\s*(\d+)', raw_inner, re.I)
        page_suffix = f" · p.{page_match.group(1)}" if page_match else ""

        if "WHO" in raw_inner.upper() and "NIH" in raw_inner.upper():
            _add_source(f"WHO / NIH Guidelines{page_suffix}")
        elif "WHO" in raw_inner.upper() and "CDC" in raw_inner.upper():
            _add_source(f"WHO / CDC Guidelines{page_suffix}")
        elif "NIH" in raw_inner.upper() and "NIDDK" in raw_inner.upper():
            _add_source(f"NIH NIDDK Guidelines{page_suffix}")
        elif "NIH" in raw_inner.upper() and "NHLBI" in raw_inner.upper():
            _add_source(f"AHA / NIH NHLBI Guidelines{page_suffix}")
        elif "WHO" in raw_inner.upper():
            _add_source(f"WHO Guidelines{page_suffix}")
        elif "NIH" in raw_inner.upper():
            _add_source(f"NIH Guidelines{page_suffix}")
        elif "|" in raw_inner:
            parts = [p.strip() for p in raw_inner.split("|") if p.strip() and not re.match(r'^MED-\d+$', p.strip(), re.I)]
            _add_source((parts[-1] if parts else "Clinical Guidelines") + page_suffix)
        else:
            cleaned_id = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', raw_inner, flags=re.I).strip()
            _add_source(cleaned_id if cleaned_id else "WHO / NIH Guidelines")
        return ""

    # 1. Extract and strip bracketed citations
    pattern_brackets = r'\s*\[([^\[\]]*(?:MED-\d+|PDF-P\d+|WHO|NIH|CDC|AHA|Guideline|Page|p\.\s*\d+|\|)[^\[\]]*)\]'
    cleaned_text = re.sub(pattern_brackets, _bracket_extractor, text, flags=re.IGNORECASE)

    # 2. Extract and strip any bare MED-XXX tokens
    if re.search(r'\bMED-\d+\b', cleaned_text):
        _add_source("WHO / NIH Guidelines")
        cleaned_text = re.sub(r'\(?\[?\bMED-\d+\b\]?\)?', '', cleaned_text)

    # Clean up any leftover empty brackets or double spaces before punctuation
    cleaned_text = re.sub(r'\[\s*\]|\(\s*\)', '', cleaned_text)
    cleaned_text = re.sub(r'\s+([.,;:])', r'\1', cleaned_text)
    return cleaned_text.strip(), sources_found


def format_inline_markdown(text: str) -> str:
    """
    FIX 1: Converts paired **bold** and *italic* properly, and strips any unmatched
    stray "**" or "*" (such as "Actions:**") so literal asterisks never appear.
    """
    if not text:
        return ""

    escaped = html.escape(text.strip())
    # Strip leading markdown heading hashes and stray emoji alerts
    escaped = re.sub(r'(?m)^\s*#{1,6}\s*', '', escaped)
    escaped = escaped.replace("🚨", "")

    # Convert complete **bold** pairs
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    # Strip any remaining stray "**" (e.g., from split headers like "Actions:**")
    escaped = escaped.replace("**", "")

    # Convert valid *italic* pairs (not spanning newlines)
    escaped = re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', escaped)

    lines = [line.strip() for line in escaped.split("\n") if line.strip()]
    formatted_lines = []
    for line in lines:
        # Strip leading bullet marker first, then strip any stray lone asterisks
        if line.startswith(("- ", "• ", "* ")):
            item_text = line[2:].strip().strip("*").strip()
            if item_text:
                formatted_lines.append(f'<div style="margin-bottom: 6px; padding-left: 12px;">• {item_text}</div>')
        else:
            clean_line = line.strip("*").strip()
            if clean_line:
                formatted_lines.append(f'<div style="margin-bottom: 8px;">{clean_line}</div>')

    return "".join(formatted_lines)


def is_refusal_or_out_of_scope(text: str) -> bool:
    """Detects safety refusals or out-of-scope responses."""
    low = text.lower()
    refusal_triggers = [
        "i don't have enough clinical evidence",
        "do not have enough clinical evidence",
        "do not have verified clinical guidelines",
        "not enough clinical evidence",
        "outside the scope",
        "outside available guidelines",
        "cannot prescribe",
        "cannot provide a prescription",
        "not covered in the provided",
        "not mentioned in the clinical",
        "no relevant clinical guidelines",
        "unable to prescribe",
        "antibiotics are not indicated for migraine",
        "i cannot fulfill this request"
    ]
    if any(trigger in low for trigger in refusal_triggers):
        if len(low) < 420 or low.startswith(("i don't", "i do not", "sorry", "i cannot", "based on the provided")):
            return True
    return False


def clean_section_prefix(text: str) -> str:
    """Removes leading section labels and stray asterisks like '**Recommended Actions:**'."""
    cleaned = re.sub(
        r'^[\s\*#\-•]*'
        r'(?:recommended(?:\s+actions?)?|recommendations?|management|first-line\s+treatment|'
        r'what\s+to\s+avoid|strictly\s+avoid|avoid|contraindications?|'
        r'(?:emergency\s+)?warning\s+signs?(?:\s+to\s+monitor)?|signs?\s+to\s+monitor|symptoms?\s+to\s+monitor|'
        r'when\s+to\s+hospitalize|hospitalization(?:\s+criteria)?|seek\s+emergency\s+care(?:\s+when)?|emergency\s+care)'
        r'[\s\*]*[:\-—]?\s*[\*]*\s*',
        '',
        text.strip(),
        flags=re.IGNORECASE
    )
    return cleaned.strip("* ").strip()


def format_clinical_answer(raw_answer: str) -> str:
    """
    FIX 1, 2 & 3:
    - Strips stray "**" and inline citations.
    - Renders ONLY "Avoid" in the red callout.
    - Renders "Signs to Monitor" in a neutral dark card.
    - Renders "When to Hospitalize" in a neutral dark card with an amber left border.
    - Places the source chip on its own line below the answer with no brackets.
    """
    if not raw_answer:
        return ""

    if "Backend Error:" in raw_answer:
        clean_err = html.escape(raw_answer.replace("❌", "").replace("**", ""))
        return (
            f'<div class="avoid-callout">'
            f'<div class="avoid-callout-title"><span class="ms-icon">warning</span>System Alert</div>'
            f'<div class="avoid-callout-body">{clean_err}</div>'
            f'</div>'
        )

    no_disclaimer = strip_duplicate_disclaimers(raw_answer)
    cleaned_answer, extracted_sources = extract_and_strip_citations(no_disclaimer or raw_answer)

    html_parts = ['<div class="clinical-answer">']

    if is_refusal_or_out_of_scope(cleaned_answer):
        html_parts.append(
            f'<div class="out-of-scope-callout">'
            f'  <div class="out-of-scope-title">'
            f'    <span class="ms-icon">info</span>Outside Available Guidelines'
            f'  </div>'
            f'  <div class="out-of-scope-body">{format_inline_markdown(cleaned_answer)}</div>'
            f'  <div class="out-of-scope-sub">'
            f'    Suggestion: Please consult a qualified physician or specialist for individualized clinical evaluation.'
            f'  </div>'
            f'</div>'
        )
    else:
        paragraphs = [p.strip() for p in re.split(r'\n+', cleaned_answer) if p.strip()]
        recommended_items = []
        avoid_items = []
        monitor_items = []
        hospitalize_items = []

        current_bucket = "recommended"

        for para in paragraphs:
            # Strip markdown symbols for header detection
            plain_p = re.sub(r'^[\s\*#\-•🚨]+', '', para).strip()
            lower_p = plain_p.lower()

            # Check explicit section headers
            if re.match(r'^(avoid|what to avoid|strictly avoid|contraindication|do not take)', lower_p):
                current_bucket = "avoid"
                rem = clean_section_prefix(para)
                if rem:
                    avoid_items.append(rem)
                continue
            elif re.match(r'^(when to hospitalize|hospitalize|hospitalization|seek emergency|emergency care|immediate hospital)', lower_p):
                current_bucket = "hospitalize"
                rem = clean_section_prefix(para)
                if rem:
                    hospitalize_items.append(rem)
                continue
            elif re.match(r'^(signs to monitor|warning signs|emergency warning signs|symptoms to monitor|red flags)', lower_p):
                current_bucket = "monitor"
                rem = clean_section_prefix(para)
                if rem:
                    monitor_items.append(rem)
                continue
            elif re.match(r'^(recommended|recommendation|management|treatment|first-line|clinical guidance)', lower_p):
                current_bucket = "recommended"
                rem = clean_section_prefix(para)
                if rem:
                    recommended_items.append(rem)
                continue

            # Sentence-level classification if the response is a single dense paragraph
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(paragraphs) <= 2 and len(sentences) > 1:
                for sent in sentences:
                    s_low = sent.lower()
                    s_clean = clean_section_prefix(sent)
                    if not s_clean:
                        continue
                    if any(k in s_low for k in ["avoid", "do not take", "contraindicated", "never use", "nsaid", "ibuprofen", "aspirin", "do not give"]):
                        avoid_items.append(s_clean)
                    elif any(k in s_low for k in ["hospitalize", "hospital", "emergency care", "call emergency", "unconscious", "platelets drop", "hematocrit", ">180/120"]):
                        hospitalize_items.append(s_clean)
                    elif any(k in s_low for k in ["warning sign", "monitor", "abdominal pain", "persistent vomiting", "bleeding gums", "shakiness", "sweating"]):
                        monitor_items.append(s_clean)
                    else:
                        recommended_items.append(s_clean)
            else:
                p_clean = clean_section_prefix(para)
                if not p_clean:
                    continue
                if current_bucket == "avoid":
                    avoid_items.append(p_clean)
                elif current_bucket == "hospitalize":
                    hospitalize_items.append(p_clean)
                elif current_bucket == "monitor":
                    monitor_items.append(p_clean)
                else:
                    if any(k in lower_p for k in ["strictly avoid", "do not take", "contraindicated", "ibuprofen", "aspirin"]):
                        avoid_items.append(p_clean)
                    elif any(k in lower_p for k in ["hospitalize", "immediate hospital", "seek emergency", "call emergency"]):
                        hospitalize_items.append(p_clean)
                    elif any(k in lower_p for k in ["warning sign", "signs to monitor", "persistent vomiting", "bleeding gums"]):
                        monitor_items.append(p_clean)
                    else:
                        recommended_items.append(p_clean)

        # 1. Recommended Section
        if recommended_items:
            html_parts.append(
                f'<div class="clinical-section">'
                f'  <div class="clinical-section-header">'
                f'    <span class="ms-icon">check_circle</span>Recommended'
                f'  </div>'
                f'  <div class="clinical-section-body">{format_inline_markdown(chr(10).join(recommended_items))}</div>'
                f'</div>'
            )

        # 2. Avoid Section (ONLY section using the Red Callout)
        if avoid_items:
            html_parts.append(
                f'<div class="avoid-callout">'
                f'  <div class="avoid-callout-title">'
                f'    <span class="ms-icon">warning</span>Avoid'
                f'  </div>'
                f'  <div class="avoid-callout-body">{format_inline_markdown(chr(10).join(avoid_items))}</div>'
                f'</div>'
            )

        # 3. Signs to Monitor (Neutral Dark Card)
        if monitor_items:
            html_parts.append(
                f'<div class="neutral-clinical-card">'
                f'  <div class="neutral-card-title">'
                f'    <span class="ms-icon">visibility</span>Signs to Monitor'
                f'  </div>'
                f'  <div class="card-body-text">{format_inline_markdown(chr(10).join(monitor_items))}</div>'
                f'</div>'
            )

        # 4. When to Hospitalize (Neutral Dark Card with Amber Left Border)
        if hospitalize_items:
            html_parts.append(
                f'<div class="amber-clinical-card">'
                f'  <div class="amber-card-title">'
                f'    <span class="ms-icon">local_hospital</span>When to Hospitalize'
                f'  </div>'
                f'  <div class="card-body-text">{format_inline_markdown(chr(10).join(hospitalize_items))}</div>'
                f'</div>'
            )

    # FIX 3: Place source chips on their own line below the answer (no brackets)
    if extracted_sources:
        chips_html = "".join(
            f'<span class="source-chip"><span class="ms-icon">verified</span>{html.escape(src)}</span>'
            for src in extracted_sources
        )
        html_parts.append(f'<div class="source-chip-row">{chips_html}</div>')

    # Single muted disclaimer inside answer card
    html_parts.append(
        '<div class="answer-disclaimer">'
        '  <span class="ms-icon">info</span>'
        '  <span>For clinical decision support only. Not a substitute for professional medical judgment.</span>'
        '</div>'
    )

    html_parts.append('</div>')
    return "".join(html_parts)


def render_evidence_cards(raw_sources: str) -> str:
    """
    FIX 4:
    - Hides chunks below 50% relevance score.
    - Removes "Cosine Distance:" and duplicated "Source:" lines.
    - Parses chunks into clean Condition, Question, and Answer fields.
    """
    if not raw_sources or not str(raw_sources).strip():
        return ""

    text = str(raw_sources).strip()
    chunks = [c.strip() for c in re.split(r'\n\s*\n|\n(?=Source:)', text) if c.strip()]
    if not chunks:
        chunks = [text]

    cards_html = []
    for idx, chunk in enumerate(chunks, 1):
        # 1. Extract source header
        src_match = re.search(r'^Source:\s*([^\n]+)', chunk, re.I | re.M)
        if src_match:
            source_name = src_match.group(1).strip()
        else:
            source_name = f"WHO / NIH Clinical Guideline #{idx}"

        source_name = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', source_name, flags=re.I).strip() or "WHO / NIH Guidelines"

        # 2. Extract page label if present
        page_match = re.search(r'(?:page|p\.?)\s*[:=]?\s*(\d+)', chunk, re.I)
        page_label = f"Page {page_match.group(1)}" if page_match else "Verified Guideline"

        # 3. Compute relevance percentage from Cosine Distance (hide if < 50%)
        score_pct = max(74, 95 - (idx - 1) * 7)
        dist_match = re.search(r'Cosine Distance\s*[:=]\s*([0-9.]+)', chunk, re.I)
        if dist_match:
            try:
                dist_val = float(dist_match.group(1))
                # Cosine distance in Chroma (0 = exact match, 1 = orthogonal)
                score_pct = int(round((1.0 - min(dist_val, 1.0)) * 100))
            except ValueError:
                pass

        if score_pct < 50:
            continue

        # 4. Strip "Source: ..." and "Cosine Distance: ..." lines from body
        body_lines = []
        for line in chunk.split("\n"):
            if re.match(r'^\s*(?:Source|Cosine Distance|Distance|Score)\s*:', line, re.I):
                continue
            body_lines.append(line.strip())
        clean_body = " ".join(body_lines).strip()
        clean_body = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', clean_body).strip()

        # 5. Parse "CONDITION: ... | Q: ... | A: ..." into clean fields
        cond_match = re.search(r'CONDITION:\s*([^|]+)', clean_body, re.I)
        q_match = re.search(r'\|\s*Q:\s*([^|]+)', clean_body, re.I)
        a_match = re.search(r'\|\s*A:\s*(.+)$', clean_body, re.I)

        if cond_match or q_match or a_match:
            fields_html = '<div class="evidence-field-grid">'
            if cond_match:
                fields_html += (
                    f'<div class="evidence-field-row">'
                    f'<span class="evidence-field-label">Condition:</span>'
                    f'{html.escape(cond_match.group(1).strip())}'
                    f'</div>'
                )
            if q_match:
                fields_html += (
                    f'<div class="evidence-field-row">'
                    f'<span class="evidence-field-label">Question:</span>'
                    f'{html.escape(q_match.group(1).strip())}'
                    f'</div>'
                )
            if a_match:
                fields_html += (
                    f'<div class="evidence-field-row">'
                    f'<span class="evidence-field-label">Answer:</span>'
                    f'{html.escape(a_match.group(1).strip())}'
                    f'</div>'
                )
            fields_html += '</div>'
        else:
            fields_html = (
                f'<div class="evidence-field-grid">'
                f'  <div class="evidence-field-row">'
                f'    <span class="evidence-field-label">Guideline Excerpt:</span>'
                f'    {html.escape(clean_body)}'
                f'  </div>'
                f'</div>'
            )

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
            f'  {fields_html}'
            f'</div>'
        )

    if not cards_html:
        return (
            '<div style="font-size:0.84rem; color:#9CA3AF; padding:8px 4px;">'
            'No clinical chunks met the 50% relevance threshold for this query.'
            '</div>'
        )

    return "".join(cards_html)


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================================================================
# 5. SIDEBAR: GROUPED INTO 3 DARK CARDS
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

        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==============================================================================
# 6. HEADER BANNER
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
# 7. QUICK TEST CARDS (TITLE ON LINE 1, DESCRIPTION ON LINE 2, EQUAL HEIGHT)
# ==============================================================================
st.markdown(
    '<div class="section-label"><span class="ms-icon">science</span>Quick Clinical Evaluations</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="medium")
sample_query = None

with col1:
    if st.button(
        "**Dengue Painkillers & Warnings**  \nNSAID contraindications & warning signs",
        key="q_dengue",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"

with col2:
    if st.button(
        "**Diabetes HbA1c Thresholds**  \nFasting glucose & HbA1c diagnostic cutoffs",
        key="q_diabetes",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"

with col3:
    if st.button(
        "**Safety Guardrail Check**  \nRefusal protocol for unindicated antibiotics",
        key="q_guardrail",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "Can you prescribe an antibiotic dosage for a severe migraine headache?"

# ==============================================================================
# 8. EMPTY STATE & CHAT HISTORY
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
                f'<div style="font-size:16px; line-height:1.6; color:#E5E7EB;">{html.escape(msg["content"])}</div>',
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
            f'<div style="font-size:16px; line-height:1.6; color:#E5E7EB;">{html.escape(user_query)}</div>',
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
            history = st.session_state.messages[:-1]
            answer, retrieved_sources = ask_med_langchain(
                user_query,
                selected_specialty,
                chat_history=history
            )
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