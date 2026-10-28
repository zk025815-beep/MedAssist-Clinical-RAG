# 🏥 MedAssist AI — Evidence-Based Clinical FAQ & Guideline RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain_%26_Native_Python-green)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?logo=google&logoColor=white)

**MedAssist AI** is a production-style **Retrieval-Augmented Generation (RAG)** healthcare assistant designed to answer clinical and patient queries using verified medical guidelines (modeled on **NIH MedlinePlus**, **WHO**, **CDC**, and **AHA** standards) alongside custom uploaded medical PDFs.

To demonstrate both foundational AI engineering and modern framework orchestration, this repository provides **two complete implementations** of the exact same clinical RAG system:
1. **`Native-Python-Version/`** — Built from scratch using pure Python chunking, `sentence-transformers`, `chromadb`, and the official `google-genai` SDK.
2. **`LangChain-Version/`** — Built using **LangChain Expression Language (LCEL)**, `PyPDFLoader`, `HuggingFaceEmbeddings`, and `langchain-chroma`.

---

## 🌟 Key Engineering Features

* **🧠 Context-Enriched Atomic Chunking:** Preserves `[Condition + Category + Patient Question + Clinical Guideline Answer]` together so symptoms, diagnostic thresholds, and medication rules never lose their disease context during vectorization.
* **🩺 ChromaDB Metadata Filtering:** Users can filter vector searches by specific medical departments (*Infectious Diseases*, *Cardiology*, *Endocrinology & Metabolic*, *Gastroenterology*, *Respiratory Medicine*, *Emergency & Environmental*, or *Uploaded Medical PDF*).
* **📄 Dynamic PDF Ingestion:** Upload multi-page medical PDFs (WHO manuals, lab reference guides, clinical protocols) directly from the Streamlit sidebar for instant chunking and indexing with page-level citations.
* **🛑 Dual-Layer Hallucination Guardrails:**
  1. **Hard Cosine Distance Cutoff (`max_distance = 0.70`):** Mathematically blocks out-of-scope or unrelated queries *before* calling the LLM API, saving tokens and guaranteeing zero hallucination on unverified topics.
  2. **Deterministic Clinical Prompting (`temperature = 0.0`):** Enforces strict adherence to retrieved context, highlights **🚨 Emergency Warning Signs**, cites exact Record IDs & Source Agencies (e.g., `[MED-001 | WHO / NIH Guidelines]`), and appends a mandatory clinical disclaimer.
* **🔬 Transparent Evidence Inspector:** Every assistant reply includes an expandable **"View Retrieved Clinical Evidence"** panel displaying the exact ChromaDB chunks, metadata, and Cosine Distance scores.

---

## 🏗️ System Architecture

```text
User Query ──► SentenceTransformer (all-MiniLM-L6-v2) ──► 384-Dim Query Vector
                                                                │
                                                                ▼
              ┌──────────────────────────────────────────────────────────────┐
              │                   ChromaDB Vector Database                   │
              │  • Cosine Distance Search (Top-K = 3)                        │
              │  • Metadata Filter: where={"category": selected_specialty}   │
              └──────────────────────────────────────────────────────────────┘
                                                                │
                                                                ▼
                                                  ┌──────────────────────────┐
                                                  │ Cosine Distance > 0.70?  │
                                                  └──────┬─────────────┬─────┘
                                                     YES │             │ NO
                                                         ▼             ▼
                                              ┌────────────────┐  ┌─────────────────────────────┐
                                              │ Block LLM Call │  │ Gemini 2.5 Flash (Temp=0.0) │
                                              │ Return Safety  │  │ + Clinical System Guardrail │
                                              │ Refusal Alert  │  │ + Source & Page Citations   │
                                              └────────────────┘  └─────────────────────────────┘
```

---

## ⚖️ Native Python vs. LangChain Comparison

| Pipeline Stage | `Native-Python-Version/` (From Scratch) | `LangChain-Version/` (Framework) |
| :--- | :--- | :--- |
| **PDF & Text Splitting** | Custom `SimpleTextSplitter` class + `pypdf.PdfReader` loop | `PyPDFLoader.load_and_split()` + `RecursiveCharacterTextSplitter` |
| **Embeddings** | `SentenceTransformer("all-MiniLM-L6-v2").encode()` | `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` |
| **Vector Store** | `chromadb.PersistentClient()` with manual ID/metadata indexing | `Chroma.from_documents()` & `similarity_search_with_score()` |
| **LLM Orchestration** | `google.genai.Client()` with `GenerateContentConfig` | LCEL Chain: `prompt \| ChatGoogleGenerativeAI \| StrOutputParser()` |
| **Engineering Goal** | Full control & transparency into low-level RAG math and data structures | Rapid, concise, industry-standard pipeline orchestration |

---

## 📂 Repository Structure

```text
MedAssist-Clinical-RAG/
│
├── Native-Python-Version/              # Implementation 1: Built from scratch
│   ├── data/
│   │   └── real_world_medical_faqs.csv # Auto-generated clinical dataset
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Clinical dataset generator & loader
│   │   ├── indexer.py                  # Custom chunker, embeddings & ChromaDB storage
│   │   └── rag_engine.py               # Vector retrieval, distance cutoff & Gemini SDK
│   ├── app.py                          # Streamlit UI (Native backend)
│   └── requirements.txt
│
├── LangChain-Version/                  # Implementation 2: Built with LangChain LCEL
│   ├── rag_chain.py                    # Compact LangChain Document, Chroma & LCEL pipeline
│   ├── app.py                          # Streamlit UI (LangChain backend)
│   └── requirements.txt
│
├── .env                                # Local environment variables (not tracked in Git)
├── .gitignore                          # Protects .env, venv/, and chroma_db/
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started (VS Code / Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MedAssist-Clinical-RAG.git
cd MedAssist-Clinical-RAG
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate
# Windows (Command Prompt):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate
```

### 3. Configure Your API Key
Create a `.env` file in the root `MedAssist-Clinical-RAG/` directory and add your free Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/)):
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

### ▶️ Option A: Run the Native Python Version
```bash
cd Native-Python-Version
pip install -r requirements.txt
streamlit run app.py
```

### ▶️ Option B: Run the LangChain Version
```bash
cd LangChain-Version
pip install -r requirements.txt
streamlit run app.py
```

Once launched, Streamlit will automatically open the application in your browser at `http://localhost:8501`.

---

## 🧪 Sample Clinical Test Cases

Use the built-in quick test buttons in the UI or try these prompts to evaluate retrieval accuracy and safety guardrails:

1. **Medication Contraindication & Red-Flag Test (In-Scope):**
   > *"Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"*
   * **Expected Behavior:** Retrieves `[MED-001]`, warns strictly against NSAIDs (Aspirin/Ibuprofen) due to fatal internal bleeding risk, recommends Paracetamol only, and highlights **🚨 Emergency Warning Signs**.
2. **Diagnostic Threshold Test (In-Scope):**
   > *"My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"*
   * **Expected Behavior:** Retrieves `[MED-003 | NIH NIDDK]` and accurately classifies the values under **Prediabetes** ($100\text{–}125\text{ mg/dL}$ Fasting / $5.7\%\text{–}6.4\%$ $\text{HbA1c}$).
3. **Out-of-Scope Hallucination Guardrail Test:**
   > *"Can you prescribe an antibiotic dosage for a severe migraine headache?"*
   * **Expected Behavior:** Triggers the **Clinical Safety Guardrail** (high Cosine Distance / absent from clinical context) and refuses to invent a prescription.

---

## ⚠️ Medical Disclaimer
*This project is built strictly for educational and AI engineering portfolio purposes. The information retrieved by MedAssist AI comes from sample clinical reference guidelines and is not a substitute for professional medical advice, diagnosis, or treatment.*