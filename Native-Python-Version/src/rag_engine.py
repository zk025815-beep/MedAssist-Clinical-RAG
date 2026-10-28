import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.indexer import embed_model, get_collection

# Load GEMINI_API_KEY from .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing! Please add it to your .env file.")

llm_client = genai.Client(api_key=api_key)

MEDICAL_SYSTEM_PROMPT = """You are MedAssist AI, an Evidence-Based Clinical FAQ Assistant.
Your role is to answer patient and healthcare queries STRICTLY using the provided [CLINICAL CONTEXT].

STRICT MEDICAL SAFETY RULES:
1. GROUNDING ONLY: Use ONLY the facts, thresholds, and medications explicitly stated in the [CLINICAL CONTEXT]. Never invent drug dosages or treatments.
2. SOURCE CITATION: Cite the exact Record ID and Source Agency at the end of key facts, e.g., [MED-001 | WHO / NIH Clinical Guidelines].
3. RED-FLAG ALERTS: If the context mentions emergency warning signs (such as internal bleeding, BP > 180/120, unconsciousness, or core temperature > 103°F), highlight them clearly in bold as **🚨 Emergency Warning Signs**.
4. OUT-OF-SCOPE REFUSAL: If the answer is not in the [CLINICAL CONTEXT], state:
   "I do not have verified clinical guidelines for this question in my current medical knowledge base. Please consult a licensed physician."
5. MANDATORY DISCLAIMER: End every medical response with a one-line italicized disclaimer:
   "*Disclaimer: This information is from clinical reference guidelines for educational purposes and is not a substitute for professional medical diagnosis.*"
"""

def query_medical_rag(user_question: str, category_filter: str = "All Specialties", top_k: int = 3, max_distance: float = 0.70):
    med_collection = get_collection()
    query_vec = embed_model.encode([user_question]).tolist()

    where_clause = None if category_filter == "All Specialties" else {"category": category_filter}

    results = med_collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        return "⚠️ No records found under this medical specialty filter.", "No chunks retrieved."

    formatted_chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        header = f"[{meta['id']} | {meta['source']} | Specialty: {meta['category']} | Distance: {dist:.3f}]"
        formatted_chunks.append(f"{header}\n{doc}")

    inspector_output = ("\n" + "=" * 55 + "\n").join(formatted_chunks)

    # Hard Distance Guardrail
    if dists[0] > max_distance:
        return (
            f"🛑 **Clinical Safety Guardrail Triggered:** Your question has a Cosine Distance of `{dists[0]:.3f}` "
            f"(exceeding the safety threshold of `{max_distance}`). I cannot provide medical information on topics "
            f"outside my verified clinical database. Please consult a healthcare professional.",
            inspector_output
        )

    prompt = f"""[CLINICAL CONTEXT]:
{inspector_output}

[PATIENT / USER QUESTION]:
{user_question}
"""
    response = llm_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=MEDICAL_SYSTEM_PROMPT,
            temperature=0.0
        )
    )
    return response.text, inspector_output