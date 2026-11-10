import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# 1. Detect pure formatting/meta requests that should NOT trigger a new DB search
FORMAT_PATTERNS = re.compile(
    r"^(make it|keep it|can you make it|please make it|rewrite|summarize|shorten|"
    r"shorter|brief|concise|simpler|bullet points|in bullets|translate|explain like|"
    r"isko chota|mukhtasir|asan الفاظ|urdu mein|in urdu|in english)\b",
    re.IGNORECASE
)

def is_formatting_followup(query: str, chat_history: list) -> bool:
    """Returns True if the user is asking to reformat/shorten the previous answer."""
    if not chat_history:
        return False
    clean_q = query.strip()
    if FORMAT_PATTERNS.search(clean_q) and len(clean_q.split()) <= 12:
        return True
    return False


def contextualize_medical_query(llm, query: str, chat_history: list) -> str:
    """
    Rewrites follow-up clinical questions (e.g., 'What about in pregnancy?')
    into standalone search queries for ChromaDB.
    """
    if not chat_history:
        return query

    recent_turns = ""
    for msg in chat_history[-4:]:
        role = "Clinician" if msg.get("role") == "user" else "Assistant"
        recent_turns += f"{role}: {msg.get('content', '')[:300]}\n"

    rewrite_prompt = [
        SystemMessage(content=(
            "Given the recent clinical conversation and a follow-up question, "
            "rewrite the follow-up question into a standalone English medical search query "
            "that includes the specific condition or drug being discussed. "
            "If the question is already standalone, return it in concise English. "
            "Return ONLY the rewritten query text, nothing else."
        )),
        HumanMessage(content=f"Conversation:\n{recent_turns}\nFollow-up: {query}\nStandalone Query:")
    ]

    try:
        rewritten = llm.invoke(rewrite_prompt).content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query


# 2. Load .env from current or parent folder
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
load_dotenv(current_dir / ".env") or load_dotenv(parent_dir / ".env") or load_dotenv()

# 3. Core NIH/WHO Clinical FAQs
CLINICAL_DOCS = [
    Document(
        page_content="CONDITION: Dengue Fever | Q: Warning signs & painkillers to avoid? | A: High fever (40°C/104°F), headache, eye pain. WARNING SIGNS: severe abdominal pain, persistent vomiting, bleeding gums/nose, blood in vomit/stool. MEDICATION RULE: Take ONLY Paracetamol (Panadol). Strictly AVOID Aspirin (Disprin), Ibuprofen (Brufen), Naproxen (NSAIDs) due to fatal internal bleeding risk.",
        metadata={"id": "MED-001", "category": "Infectious Diseases", "source": "WHO / NIH Guidelines"}
    ),
    Document(
        page_content="CONDITION: Dengue Fever | Q: Home management & hospitalization? | A: Bed rest, ORS, fluids, Paracetamol (max 4g/day for adults). Hospitalize immediately if platelets drop rapidly with warning signs, hematocrit rises sharply, or patient cannot drink fluids.",
        metadata={"id": "MED-002", "category": "Infectious Diseases", "source": "WHO / NIH Guidelines"}
    ),
    Document(
        page_content="CONDITION: Type 2 Diabetes | Q: Diagnostic blood sugar thresholds (Fasting & HbA1c)? | A: Normal: Fasting <100 mg/dL, HbA1c <5.7%. Prediabetes: Fasting 100–125 mg/dL or HbA1c 5.7%–6.4%. Diabetes: Fasting >=126 mg/dL, HbA1c >=6.5%, or 2-hr OGTT >=200 mg/dL.",
        metadata={"id": "MED-003", "category": "Endocrinology & Metabolic", "source": "NIH NIDDK"}
    ),
    Document(
        page_content="CONDITION: Hypoglycemia (Low Blood Sugar) | Q: 15-15 Rule treatment? | A: Below 70 mg/dL causes shakiness, sweating, dizziness. 15-15 Rule: Take 15g fast-acting carbs (half cup juice, 1 tbsp sugar/honey, or 3-4 glucose tabs), wait 15 mins, recheck. Repeat if <70 mg/dL. If unconscious (behosh), give Glucagon injection, NOT oral food/drink.",
        metadata={"id": "MED-004", "category": "Endocrinology & Metabolic", "source": "NIH NIDDK"}
    ),
    Document(
        page_content="CONDITION: Hypertension | Q: Blood pressure categories & Hypertensive Crisis? | A: Normal: <120/80. Elevated: 120-129/<80. Stage 1: 130-139/80-89. Stage 2: >=140/90. HYPERTENSIVE CRISIS (Emergency): >180/120 mmHg—seek immediate hospital care if chest pain, shortness of breath, or numbness occurs.",
        metadata={"id": "MED-005", "category": "Cardiology", "source": "AHA / NIH NHLBI"}
    ),
    Document(
        page_content="CONDITION: Acute Gastroenteritis | Q: Antibiotics & ORS for diarrhea? | A: Most watery diarrhea is viral and does NOT need antibiotics. Antibiotics only for bloody diarrhea (dysentery) or severe Cholera. Treat with Low-Osmolarity ORS in 1L boiled water + Zinc (20mg daily for 10-14 days in children).",
        metadata={"id": "MED-006", "category": "Gastroenterology", "source": "WHO / CDC"}
    ),
    Document(
        page_content="CONDITION: Bronchial Asthma | Q: Rescue vs Controller Inhaler? | A: Rescue Inhalers (Salbutamol/Albuterol) relax airways fast during acute attacks. Controller Inhalers (Budesonide/Fluticasone) are taken daily to prevent long-term inflammation. Using rescue inhaler >2x/week means poor control.",
        metadata={"id": "MED-007", "category": "Respiratory Medicine", "source": "NIH NHLBI"}
    ),
    Document(
        page_content="CONDITION: Heat Exhaustion vs Heatstroke | Q: Differences & First Aid? | A: Heat Exhaustion: heavy sweating, cold clammy skin, weak pulse (cool room + sips of water). HEATSTROKE (Emergency): core temp >103°F, hot red skin, confusion/unconsciousness. Call emergency, cool with ice packs, do NOT give fluids.",
        metadata={"id": "MED-008", "category": "Emergency & Environmental", "source": "CDC / NIH"}
    )
]

# 4. Initialize Multilingual Embeddings & ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma.from_documents(
    documents=CLINICAL_DOCS,
    embedding=embeddings,
    collection_name="med_multilingual_kb",
    collection_metadata={"hnsw:space": "cosine"}
)

# 5. Initialize OpenRouter LLM
openrouter_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_key:
    raise ValueError(
        "❌ OPENROUTER_API_KEY nahi mili! Apni .env file mein OPENROUTER_API_KEY=sk-or-v1-... add karein."
    )

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0
)

# 6. LangChain Prompts & Chains (Includes Chat History Support)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are MedAssist AI. Answer STRICTLY using the [CLINICAL CONTEXT] and [RECENT CONVERSATION] below.
1. LANGUAGE MATCHING: If the user asks in Roman Urdu or Urdu, reply clearly in Roman Urdu! If they ask in English, reply in English.
2. Cite the Record ID & Source (e.g., [MED-004 | NIH NIDDK]).
3. Structure your answer clearly into Recommended actions, what to Avoid (if applicable), and Emergency Warning Signs (if applicable).
4. If the answer is not in the context, say: "I do not have verified clinical guidelines for this question. Please consult a physician." """),
    ("human", "[RECENT CONVERSATION]:\n{chat_history}\n\n[CLINICAL CONTEXT]:\n{context}\n\n[QUESTION]:\n{question}")
])

query_translator = (
    ChatPromptTemplate.from_template(
        "Translate the following medical question into concise English medical search keywords. "
        "If it is already in English, return it unchanged. Return ONLY the English query:\n{question}"
    )
    | llm
    | StrOutputParser()
)

rag_chain = prompt | llm | StrOutputParser()

# 7. PDF Uploader Function
def index_uploaded_pdfs(pdf_paths: list[str]) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=650, chunk_overlap=100)
    new_docs = []
    for path in pdf_paths:
        pages = PyPDFLoader(path).load_and_split(splitter)
        for p in pages:
            p.metadata = {
                "id": f"PDF-P{p.metadata.get('page', 0)+1}",
                "category": "Uploaded Medical PDF",
                "source": f"{os.path.basename(path)} (Page {p.metadata.get('page', 0)+1})"
            }
        new_docs.extend(pages)
    vectorstore.add_documents(new_docs)
    return f"Added {len(new_docs)} chunks from {len(pdf_paths)} PDF(s) via LangChain!"

# 8. Main Query Function (Fixed: Always returns a 2-tuple `(answer, sources)`)
def ask_med_langchain(user_query: str, selected_specialty: str = "All Specialties", chat_history: list = None):
    if chat_history is None:
        chat_history = []

    # Find the last assistant response & its sources (if any)
    last_assistant_msg = next(
        (m for m in reversed(chat_history) if m.get("role") == "assistant"),
        None
    )

    # =========================================================================
    # PATH A: DIRECT REWRITE / SHORTEN (NO NEW CHROMADB SEARCH)
    # =========================================================================
    if is_formatting_followup(user_query, chat_history) and last_assistant_msg:
        rewrite_messages = [
            SystemMessage(content=(
                "You are MedAssist AI. Modify the previous clinical response strictly "
                "according to the user's formatting instruction (e.g., make it shorter, "
                "use bullet points). Preserve all clinical accuracy, safety warnings, "
                "and existing source tags like [MED-001 | WHO / NIH Guidelines]. Do not add new medical claims."
            )),
            AIMessage(content=last_assistant_msg["content"]),
            HumanMessage(content=f"Instruction: {user_query}")
        ]
        modified_answer = llm.invoke(rewrite_messages).content
        return modified_answer, last_assistant_msg.get("sources")

    # =========================================================================
    # PATH B: CLINICAL QUESTION OR FOLLOW-UP (CONTEXTUALIZE -> RETRIEVE -> ANSWER)
    # =========================================================================
    # 1. Contextualize follow-up query with chat history, then ensure English keywords for search
    if chat_history:
        search_query = contextualize_medical_query(llm, user_query, chat_history)
    else:
        try:
            search_query = query_translator.invoke({"question": user_query}).strip()
        except Exception:
            search_query = user_query

    # 2. Query ChromaDB with optional Specialty Metadata Filter
    filter_dict = None if selected_specialty == "All Specialties" else {"category": selected_specialty}
    results_with_scores = vectorstore.similarity_search_with_score(
        search_query,
        k=3,
        filter=filter_dict
    )

    if not results_with_scores:
        return (
            "I do not have verified clinical guidelines for this question in the selected specialty. Please consult a physician.",
            None
        )

    # 3. Format context for LLM and evidence string for UI cards
    context_blocks = []
    source_blocks = []
    for doc, score in results_with_scores:
        doc_id = doc.metadata.get("id", "REF")
        doc_source = doc.metadata.get("source", "WHO / NIH Guidelines")
        doc_cat = doc.metadata.get("category", "General")
        context_blocks.append(f"[{doc_id} | {doc_source} | Category: {doc_cat}]\n{doc.page_content}")
        source_blocks.append(
            f"Source: {doc_source} ({doc_cat})\n"
            f"Cosine Distance: {score:.4f}\n"
            f"{doc.page_content}"
        )

    context_str = "\n\n".join(context_blocks)
    retrieved_sources_str = "\n\n".join(source_blocks)

    # 4. Format recent conversation history for prompt
    history_block = "No prior messages."
    if chat_history:
        lines = []
        for msg in chat_history[-4:]:
            role = "Clinician" if msg.get("role") == "user" else "MedAssist AI"
            lines.append(f"{role}: {msg.get('content', '')[:400]}")
        history_block = "\n".join(lines)

    # 5. Generate final clinical response & return 2-tuple
    answer = rag_chain.invoke({
        "chat_history": history_block,
        "context": context_str,
        "question": user_query
    })

    return answer, retrieved_sources_str