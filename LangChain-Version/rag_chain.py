import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Core NIH/WHO Clinical FAQs converted directly into LangChain 'Document' objects
CLINICAL_DOCS = [
    Document(
        page_content="CONDITION: Dengue Fever | Q: Warning signs & painkillers to avoid? | A: High fever (40°C/104°F), headache, eye pain. WARNING SIGNS: severe abdominal pain, persistent vomiting, bleeding gums/nose, blood in vomit/stool. MEDICATION RULE: Take ONLY Paracetamol. Strictly AVOID Aspirin, Ibuprofen, Naproxen (NSAIDs) due to fatal internal bleeding risk.",
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
        page_content="CONDITION: Hypoglycemia (Low Blood Sugar) | Q: 15-15 Rule treatment? | A: Below 70 mg/dL causes shakiness, sweating, dizziness. 15-15 Rule: Take 15g fast-acting carbs (half cup juice, 1 tbsp sugar/honey, or 3-4 glucose tabs), wait 15 mins, recheck. Repeat if <70 mg/dL. If unconscious, give Glucagon injection, NOT oral food/drink.",
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

# 2. Initialize LangChain Embeddings, VectorStore & LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=CLINICAL_DOCS,
    embedding=embeddings,
    collection_name="med_langchain_kb",
    collection_metadata={"hnsw:space": "cosine"}
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

# 3. LangChain Prompt Template & LCEL Chain (prompt | llm | StrOutputParser)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are MedAssist AI. Answer STRICTLY using the [CLINICAL CONTEXT] below.
1. Cite the Record ID & Source (e.g., [MED-001 | WHO / NIH Guidelines]).
2. Highlight emergency symptoms in bold as **🚨 Emergency Warning Signs**.
3. If the answer is not in the context, say: "I do not have verified clinical guidelines for this question. Please consult a physician."
4. End with: *Disclaimer: For educational purposes only; not a substitute for professional medical advice.*"""),
    ("human", "[CLINICAL CONTEXT]:\n{context}\n\n[QUESTION]:\n{question}")
])

rag_chain = prompt | llm | StrOutputParser()

def index_uploaded_pdfs(pdf_paths: list[str]) -> str:
    """Uses LangChain's PyPDFLoader and RecursiveCharacterTextSplitter in 5 lines!"""
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
    return f"✅ Added {len(new_docs)} chunks from {len(pdf_paths)} PDF(s) via LangChain!"

def ask_med_langchain(question: str, specialty: str = "All Specialties", max_distance: float = 0.70):
    """Retrieves via LangChain Chroma with score thresholding & invokes the LCEL chain."""
    filter_dict = None if specialty == "All Specialties" else {"category": specialty}
    
    # One-line similarity search with score in LangChain!
    results = vectorstore.similarity_search_with_score(question, k=3, filter=filter_dict)
    if not results:
        return "⚠️ No records found for this specialty.", "No chunks retrieved."

    inspector = "\n\n".join([
        f"[{doc.metadata['id']} | {doc.metadata['source']} | Distance: {score:.3f}]\n{doc.page_content}"
        for doc, score in results
    ])

    best_score = results[0][1]
    if best_score > max_distance:
        return f"🛑 **Safety Guardrail Triggered:** Distance `{best_score:.3f}` > `{max_distance}`. Outside clinical knowledge base.", inspector

    # Invoke the LangChain pipeline!
    answer = rag_chain.invoke({"context": inspector, "question": question})
    return answer, inspector