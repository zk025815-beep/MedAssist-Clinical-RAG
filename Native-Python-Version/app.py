import os
import tempfile
import traceback
import streamlit as st
from src.data_loader import load_or_create_medical_df
from src.indexer import index_medical_data
from src.rag_engine import query_medical_rag

# 1. Page Configuration
st.set_page_config(
    page_title="MedAssist AI — Native Python RAG",
    page_icon="🏥",
    layout="wide"
)

# 2. Cache Initial Data Load & ChromaDB Indexing (Runs ONLY ONCE on startup!)
@st.cache_resource
def initialize_knowledge_base():
    df = load_or_create_medical_df()
    status = index_medical_data()
    return df, status

df_med, init_status = initialize_knowledge_base()
specialties_list = (
    ["All Specialties"]
    + sorted(df_med["category"].unique().tolist())
    + ["Uploaded Medical PDF"]
)

# 3. Sidebar: Specialty Filter & PDF Uploader
with st.sidebar:
    st.title("🩺 Clinical Controls")
    st.caption("Architecture: **Native Python + ChromaDB**")
    st.success(init_status)

    selected_specialty = st.selectbox(
        "Filter by Medical Specialty:",
        options=specialties_list,
        index=0
    )

    st.divider()
    st.subheader("📄 Upload Medical PDFs")
    uploaded_files = st.file_uploader(
        "Upload WHO Guides / Lab Manuals (.pdf)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("➕ Index PDFs into ChromaDB", use_container_width=True):
        if not uploaded_files:
            st.warning("⚠️ Please select at least one PDF file first.")
        else:
            with st.spinner("Extracting pages, chunking & embedding into ChromaDB..."):
                temp_paths = []
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    file_path = os.path.join(temp_dir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(file_path)

                upload_status = index_medical_data(extra_files=temp_paths)
                st.success(upload_status)

    st.divider()
    if st.button("🗑️ Clear Consultation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. Main Header & Quick Clinical Test Cases
st.title("🏥 MedAssist AI — Clinical FAQ & Guideline Assistant (Native Python)")
st.caption(
    "Built from scratch using **Pure Python Chunking**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, "
    "**ChromaDB Metadata Filtering**, and **Gemini 2.5 Flash**."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("**🧪 Quick Clinical Test Cases (Click to Run):**")
col1, col2, col3 = st.columns(3)
sample_query = None

with col1:
    if st.button("🦟 Dengue Painkillers & Warnings", use_container_width=True):
        sample_query = "Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"
with col2:
    if st.button("🩸 Diabetes HbA1c Thresholds", use_container_width=True):
        sample_query = "My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"
with col3:
    if st.button("🛑 Guardrail Test (Migraine)", use_container_width=True):
        sample_query = "Can you prescribe an antibiotic dosage for a severe migraine headache?"

st.divider()

# 5. Render Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔬 View Retrieved Clinical Evidence (ChromaDB & Cosine Distance)"):
                st.code(msg["sources"], language="text")

# 6. Handle New Question
chat_input = st.chat_input("Ask a clinical question (e.g., What is the difference between a rescue and controller inhaler?)...")
user_query = sample_query or chat_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching ChromaDB vectors & verifying clinical guidelines..."):
            try:
                reply, sources = query_medical_rag(
                    user_question=user_query,
                    category_filter=selected_specialty
                )
            except Exception as e:
                reply = f"❌ **Backend Error:** `{str(e)}`"
                sources = traceback.format_exc()

        st.markdown(reply)
        if sources:
            with st.expander("🔬 View Retrieved Clinical Evidence (ChromaDB & Cosine Distance)"):
                st.code(sources, language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources
    })