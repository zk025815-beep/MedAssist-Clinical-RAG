import os
import tempfile
import streamlit as st
from rag_chain import CLINICAL_DOCS, ask_med_langchain, index_uploaded_pdfs

# 1. Page Configuration
st.set_page_config(
    page_title="MedAssist AI — Clinical RAG",
    page_icon="🏥",
    layout="wide"
)

# 2. Sidebar: Specialty Filter & PDF Uploader
with st.sidebar:
    st.title("🩺 Clinical Controls")
    st.markdown("Filter vector search by specialty or upload custom medical PDFs.")

    specialties = (
        ["All Specialties"]
        + sorted(list({d.metadata["category"] for d in CLINICAL_DOCS}))
        + ["Uploaded Medical PDF"]
    )
    selected_specialty = st.selectbox(
        "Filter by Medical Specialty:",
        options=specialties,
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
            with st.spinner("Chunking and indexing PDF(s)..."):
                temp_paths = []
                temp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    # Save uploaded Streamlit file to a temporary path with its real filename
                    file_path = os.path.join(temp_dir, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(file_path)

                status_msg = index_uploaded_pdfs(temp_paths)
                st.success(status_msg)

    st.divider()
    if st.button("🗑️ Clear Consultation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 3. Main Header & Quick Test Buttons
st.title("🏥 MedAssist AI — Clinical FAQ & Guideline Assistant")
st.caption(
    "Powered by **NIH / WHO Clinical FAQs**, **LangChain**, **ChromaDB Metadata Filtering**, "
    "and **Gemini 2.5 Flash** with strict non-hallucination guardrails."
)

# Initialize chat history in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Example Prompts at the top
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

# 4. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this assistant message has retrieved sources, show them in a clean dropdown!
        if msg.get("sources"):
            with st.expander("🔬 View Retrieved Clinical Evidence (ChromaDB & Cosine Distance)"):
                st.code(msg["sources"], language="text")

# 5. Handle New User Input (from chat input bar OR quick test buttons)
chat_input = st.chat_input("Ask a clinical question (e.g., What is the 15-15 rule for hypoglycemia?)...")
user_query = sample_query or chat_input

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching clinical database & verifying guidelines..."):
            try:
                answer, retrieved_sources = ask_med_langchain(user_query, selected_specialty)
            except Exception as e:
                answer = f"❌ **Backend Error:** `{str(e)}`"
                retrieved_sources = None

        st.markdown(answer)
        if retrieved_sources:
            with st.expander("🔬 View Retrieved Clinical Evidence (ChromaDB & Cosine Distance)"):
                st.code(retrieved_sources, language="text")

    # Save assistant response + sources to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_sources
    })