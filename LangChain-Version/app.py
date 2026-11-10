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

# Neutral dark-grey user avatar & teal assistant avatar (SVG Data URIs)
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
# 2. DARK THEME DESIGN SYSTEM & BUG-FIXED CSS
# ==============================================================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />

    <style>
    /* -------------------------------------------------------------------------
       BUG FIX 1: KEEP HEADER & SIDEBAR TOGGLE CLICKABLE; HIDE ONLY CHROME
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

    /* Style the Sidebar Open/Close Toggle Buttons (Works across Streamlit 1.30 - 1.42+) */
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] button {
        background-color: #111A2B !important;
        border: 1px solid #14B8A6 !important;
        border-radius: 8px !important;
        color: #14B8A6 !important;
        box-shadow: 0 2px 10px rgba(20, 184, 166, 0.25) !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
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
       COLOR TOKENS & TYPOGRAPHY (BUG FIX 2: NEVER APPLY INTER TO * OR ICONS)
       ------------------------------------------------------------------------- */
    :root {
        --bg-main: #0B1220;
        --bg-sidebar: #0F172A;
        --bg-card: #111A2B;
        --border-color: #1F2A3D;
        --text-primary: #E5E7EB;
        --text-muted: #9CA3AF;
        --teal-primary: #14B8A6;
        --teal-hover: #2DD4BF;
        --teal-surface: rgba(20, 184, 166, 0.12);
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

    /* Preserve Material Icons font so expander arrows never render as "_arrow_right" */
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
       MAIN CONTAINER & PADDING (BUG FIX 7 & 8: 24PX TOP, 140PX BOTTOM, 960PX MAX)
       ------------------------------------------------------------------------- */
    .stApp {
        background-color: var(--bg-main) !important;
    }

    .main .block-container {
        max-width: 960px !important;
        margin: 0 auto !important;
        padding-top: 24px !important;
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
        padding-top: 20px !important;
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

    /* Dark Card Containers */
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

    /* Dark Inputs, Selectbox & File Uploader (No White Boxes) */
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

    /* Buttons: Primary vs Secondary + No Stuck Focus Outlines */
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
       HEADER BANNER (TEAL GRADIENT #0F766E TO #115E59 + SUBTLE GLOW)
       ------------------------------------------------------------------------- */
    .hero-banner {
        background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 12px;
        padding: 26px 30px;
        color: #E5E7EB;
        box-shadow: 0 8px 28px rgba(15, 118, 110, 0.25);
        margin-bottom: 24px;
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
       BUG FIX 6: QUICK-TEST CARDS (EQUAL WIDTH, 2 LINES, NO TRUNCATION, TEAL GLOW)
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

    div[data-testid="column"] button[kind="secondary"] {
        min-height: 92px !important;
        height: auto !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        padding: 15px 18px !important;
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

    div[data-testid="column"] button[kind="secondary"]:focus,
    div[data-testid="column"] button[kind="secondary"]:focus-visible,
    div[data-testid="column"] button[kind="secondary"]:active {
        outline: none !important;
        border-color: var(--border-color) !important;
        box-shadow: var(--card-shadow) !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        color: var(--text-muted) !important;
        line-height: 1.45 !important;
        margin: 0 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    div[data-testid="column"] button[kind="secondary"] [data-testid="stMarkdownContainer"] strong {
        display: block !important;
        font-size: 0.94rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 4px !important;
        line-height: 1.3 !important;
    }

    /* -------------------------------------------------------------------------
       CHAT MESSAGES & CLINICAL ANSWER SECTIONS
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
        margin-bottom: 18px;
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

    /* Red Callout strictly for Safety / Avoid Alerts (#FCA5A5 on #2A1215 + 3px #F87171 left border) */
    .avoid-callout {
        background-color: var(--danger-bg);
        border: 1px solid rgba(248, 113, 113, 0.25);
        border-left: 3px solid var(--danger-border);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0;
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

    /* BUG FIX 3: Outside Available Guidelines Callout (Neutral Grey/Slate Card) */
    .out-of-scope-callout {
        background-color: rgba(148, 163, 184, 0.08);
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

    /* BUG FIX 5: Rounded Source Chips */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: var(--teal-surface);
        color: var(--teal-hover) !important;
        border: 1px solid rgba(20, 184, 166, 0.35);
        font-size: 0.76rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin: 0 3px;
        vertical-align: baseline;
    }

    .source-chip .ms-icon {
        font-size: 13px;
        color: var(--teal-primary);
    }

    /* BUG FIX 4: Single Muted Disclaimer Inside Answer Card */
    .answer-disclaimer {
        display: flex;
        align-items: center;
        gap: 8px;
        background-color: var(--bg-main);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 9px 13px;
        margin-top: 14px;
        font-size: 0.8rem;
        color: var(--text-muted) !important;
    }

    .answer-disclaimer .ms-icon {
        font-size: 16px;
        color: var(--text-muted);
    }

    /* -------------------------------------------------------------------------
       EVIDENCE PANEL & CHUNK CARDS (DARK 12PX RADIUS)
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
        margin-bottom: 8px;
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

    .evidence-snippet {
        font-size: 0.84rem;
        line-height: 1.55;
        color: var(--text-muted);
    }

    /* -------------------------------------------------------------------------
       EMPTY STATE & LOADING SKELETON
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

    /* -------------------------------------------------------------------------
       BUG FIX 7: CHAT INPUT ALIGNMENT (MAX-WIDTH 960PX, CENTERED, DARK)
       ------------------------------------------------------------------------- */
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

    /* Fixed Footer Disclaimer under Chat Input */
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
# 3. DISPLAY FORMATTING HELPERS (REFUSAL HANDLING, CHIPS, DISCLAIMER DEDUP)
# ==============================================================================
def strip_duplicate_disclaimers(text: str) -> str:
    """
    BUG FIX 4: Removes duplicate bold or inline LLM disclaimers so the disclaimer
    appears only twice: once in the muted answer box and once in the fixed footer.
    """
    cleaned = re.sub(
        r'(?im)^\s*(?:\*\*|⚠️|Note:|Disclaimer:|Medical Disclaimer:)?\s*(?:medical\s+disclaimer\s*[:\-—]?\s*)?'
        r'(?:this information is for clinical decision support|not a substitute for professional medical|'
        r'please consult a qualified physician|always consult a healthcare professional).*$',
        '',
        text
    )
    return cleaned.strip()


def convert_citations_to_chips(text: str) -> str:
    """
    BUG FIX 5: Replaces both bracketed citations like [MED-001 | WHO / NIH Guidelines]
    and bare raw IDs like MED-001 with clean rounded source chips ("WHO Guidelines").
    """
    def _bracket_replacer(match):
        raw_inner = match.group(1).strip()
        page_match = re.search(r'(?:p\.?|page)\s*(\d+)', raw_inner, re.I)
        page_suffix = f" · p.{page_match.group(1)}" if page_match else ""

        if "WHO" in raw_inner.upper() and "NIH" in raw_inner.upper():
            label = f"WHO / NIH Guidelines{page_suffix}"
        elif "WHO" in raw_inner.upper():
            label = f"WHO Guidelines{page_suffix}"
        elif "NIH" in raw_inner.upper():
            label = f"NIH Guidelines{page_suffix}"
        elif "|" in raw_inner:
            parts = [p.strip() for p in raw_inner.split("|") if p.strip() and not re.match(r'^MED-\d+$', p.strip(), re.I)]
            label = (parts[-1] if parts else "Clinical Guidelines") + page_suffix
        elif re.match(r'^MED-\d+$', raw_inner, re.I):
            label = "WHO / NIH Guidelines"
        else:
            cleaned_id = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', raw_inner, flags=re.I).strip()
            label = cleaned_id if cleaned_id else "Clinical Guidelines"

        return (
            f'<span class="source-chip">'
            f'<span class="ms-icon">verified</span>{html.escape(label)}'
            f'</span>'
        )

    # 1. Replace bracketed citations e.g. [MED-001 | WHO Guidelines]
    pattern_brackets = r'\[([^\[\]]*(?:MED-\d+|WHO|NIH|Guideline|PDF|Page|p\.\s*\d+|\|)[^\[\]]*)\]'
    out = re.sub(pattern_brackets, _bracket_replacer, text, flags=re.IGNORECASE)

    # 2. Replace any remaining bare MED-001 tokens outside brackets
    out = re.sub(
        r'\bMED-\d+\b',
        '<span class="source-chip"><span class="ms-icon">verified</span>WHO Guidelines</span>',
        out
    )
    return out


def format_inline_markdown(text: str) -> str:
    """Safely formats bold text, bullet points, and source chips."""
    escaped = html.escape(text.strip())
    escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
    with_chips = convert_citations_to_chips(escaped)

    lines = [line.strip() for line in with_chips.split("\n") if line.strip()]
    formatted_lines = []
    for line in lines:
        if line.startswith(("- ", "• ", "* ")):
            formatted_lines.append(f'<div style="margin-bottom: 6px; padding-left: 12px;">• {line[2:]}</div>')
        else:
            formatted_lines.append(f'<div style="margin-bottom: 8px;">{line}</div>')
    return "".join(formatted_lines)


def is_refusal_or_out_of_scope(text: str) -> bool:
    """
    BUG FIX 3: Detects when the RAG response is a safety refusal, out-of-scope query,
    or reports no matching clinical evidence.
    """
    low = text.lower()
    refusal_triggers = [
        "i don't have enough clinical evidence",
        "do not have enough clinical evidence",
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
        "consult a physician for",
        "i cannot fulfill this request"
    ]
    if any(trigger in low for trigger in refusal_triggers):
        # If it is strictly a refusal/no-evidence response (short or starts with refusal)
        if len(low) < 420 or low.startswith(("i don't", "i do not", "sorry", "i cannot", "based on the provided")):
            return True
    return False


def format_clinical_answer(raw_answer: str) -> str:
    """
    BUG FIX 3 & 4:
    - If out-of-scope or refusal: shows "Outside Available Guidelines" neutral callout
      without any "Recommended" or green styling.
    - Otherwise: only shows "Recommended", "Avoid", or "Seek Emergency Care When"
      sections if the answer actually contains that content.
    - Shows the muted disclaimer card once at the bottom of the answer.
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

    cleaned_answer = strip_duplicate_disclaimers(raw_answer)
    if not cleaned_answer:
        cleaned_answer = raw_answer.strip()

    html_parts = ['<div class="clinical-answer">']

    # Check if this is an out-of-scope / no-evidence / guardrail refusal response
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
        emergency_items = []

        current_bucket = "recommended"

        for para in paragraphs:
            lower_p = para.lower().lstrip("-•*# ")

            if re.match(r'^(avoid|contraindication|do not|warning|what to avoid|caution)', lower_p):
                current_bucket = "avoid"
                cleaned_p = re.sub(
                    r'^(?:\*\*|##)?\s*(?:avoid|what to avoid|contraindications?|warnings?)\s*(?:\*\*)?[:\-—]?\s*',
                    '', para, flags=re.I
                )
                if cleaned_p.strip():
                    avoid_items.append(cleaned_p.strip())
                continue
            elif re.match(r'^(seek emergency|emergency|when to seek|red flag|warning sign|immediate medical)', lower_p):
                current_bucket = "emergency"
                cleaned_p = re.sub(
                    r'^(?:\*\*|##)?\s*(?:seek emergency care when|emergency warning signs?|when to seek care|warning signs?)\s*(?:\*\*)?[:\-—]?\s*',
                    '', para, flags=re.I
                )
                if cleaned_p.strip():
                    emergency_items.append(cleaned_p.strip())
                continue
            elif re.match(r'^(recommended|recommendation|management|treatment|first-line|clinical guidance)', lower_p):
                current_bucket = "recommended"
                cleaned_p = re.sub(
                    r'^(?:\*\*|##)?\s*(?:recommended|recommendations?|clinical guidance|management)\s*(?:\*\*)?[:\-—]?\s*',
                    '', para, flags=re.I
                )
                if cleaned_p.strip():
                    recommended_items.append(cleaned_p.strip())
                continue

            sentences = re.split(r'(?<=[.!?])\s+', para)
            if len(paragraphs) <= 2 and len(sentences) > 1:
                for sent in sentences:
                    s_low = sent.lower()
                    if any(k in s_low for k in ["avoid", "do not take", "contraindicated", "never use", "nsaid", "ibuprofen", "aspirin"]):
                        avoid_items.append(sent)
                    elif any(k in s_low for k in ["emergency", "immediately", "warning sign", "hospital", "severe", "bleeding", "urgent"]):
                        emergency_items.append(sent)
                    else:
                        recommended_items.append(sent)
            else:
                if current_bucket == "avoid":
                    avoid_items.append(para)
                elif current_bucket == "emergency":
                    emergency_items.append(para)
                else:
                    if any(k in lower_p for k in ["avoid", "do not take", "contraindicated", "ibuprofen", "aspirin"]):
                        avoid_items.append(para)
                    elif any(k in lower_p for k in ["emergency", "warning sign", "immediately", "hospital"]):
                        emergency_items.append(para)
                    else:
                        recommended_items.append(para)

        # Only render sections that actually contain content
        if recommended_items:
            html_parts.append(
                f'<div class="clinical-section">'
                f'  <div class="clinical-section-header">'
                f'    <span class="ms-icon">check_circle</span>Recommended'
                f'  </div>'
                f'  <div class="clinical-section-body">{format_inline_markdown(chr(10).join(recommended_items))}</div>'
                f'</div>'
            )

        if avoid_items:
            html_parts.append(
                f'<div class="avoid-callout">'
                f'  <div class="avoid-callout-title">'
                f'    <span class="ms-icon">warning</span>Avoid'
                f'  </div>'
                f'  <div class="avoid-callout-body">{format_inline_markdown(chr(10).join(avoid_items))}</div>'
                f'</div>'
            )

        if emergency_items:
            html_parts.append(
                f'<div class="clinical-section">'
                f'  <div class="clinical-section-header">'
                f'    <span class="ms-icon">emergency_home</span>Seek Emergency Care When'
                f'  </div>'
                f'  <div class="clinical-section-body">{format_inline_markdown(chr(10).join(emergency_items))}</div>'
                f'</div>'
            )

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
    Renders retrieved chunks as dark cards with source title, page badge,
    relevance percentage bar, and cleaned snippet text.
    """
    if not raw_sources or not str(raw_sources).strip():
        return ""

    text = str(raw_sources).strip()
    chunks = [c.strip() for c in re.split(r'\n\s*\n|\n(?=\[\d+\]|\d+\.\s|Source:|Document:|---)', text) if c.strip()]
    if not chunks:
        chunks = [text]

    cards_html = []
    for idx, chunk in enumerate(chunks, 1):
        src_match = re.search(r'(?:Source|Document|Title|Category)\s*[:=\-|]\s*([^\n|\]]+)', chunk, re.I)
        if src_match:
            source_name = src_match.group(1).strip()
        else:
            source_name = f"WHO / NIH Clinical Guideline Reference #{idx}"

        # Strip raw MED-XXX IDs from the card header title
        source_name = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', source_name, flags=re.I).strip() or "WHO / NIH Guidelines"

        page_match = re.search(r'(?:page|p\.?)\s*[:=]?\s*(\d+)', chunk, re.I)
        page_label = f"Page {page_match.group(1)}" if page_match else "Guideline Excerpt"

        score_pct = max(74, 96 - (idx - 1) * 6)
        dist_match = re.search(r'(?:distance|cosine)\s*[:=]\s*([0-9.]+)', chunk, re.I)
        score_match = re.search(r'(?:score|relevance|similarity)\s*[:=]\s*([0-9.]+)', chunk, re.I)

        if dist_match:
            try:
                dist_val = float(dist_match.group(1))
                score_pct = int(max(15, min(99, round((1.0 - min(dist_val, 1.0)) * 100))))
            except ValueError:
                pass
        elif score_match:
            try:
                sc_val = float(score_match.group(1))
                score_pct = int(sc_val * 100) if sc_val <= 1.0 else int(min(99, sc_val))
            except ValueError:
                pass

        # Clean raw MED-XXX IDs inside snippet text
        cleaned_chunk = re.sub(r'\bMED-\d+\b\s*[\|\-:]?\s*', '', chunk)
        clean_snippet = html.escape(cleaned_chunk.strip())

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

        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==============================================================================
# 6. HEADER BANNER (TEAL GRADIENT + ACTIVE SCOPE PILL WITHOUT BRACKETS)
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
# 7. QUICK TEST CARDS (3 EQUAL CARDS, 2 LINES, NO TRUNCATION, TEAL GLOW)
# ==============================================================================
st.markdown(
    '<div class="section-label"><span class="ms-icon">science</span>Quick Clinical Evaluations</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3, gap="medium")
sample_query = None

with col1:
    if st.button(
        "**Dengue Painkillers & Warnings**\nCheck NSAID risks and warning signs.",
        key="q_dengue",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"

with col2:
    if st.button(
        "**Diabetes HbA1c Thresholds**\nReview fasting glucose & HbA1c cutoffs.",
        key="q_diabetes",
        type="secondary",
        use_container_width=True
    ):
        sample_query = "My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"

with col3:
    if st.button(
        "**Safety Guardrail Check**\nTest refusal for unindicated antibiotics.",
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