import traceback
import gradio as gr
from src.data_loader import load_or_create_medical_df
from src.indexer import index_medical_data
from src.rag_engine import query_medical_rag

# Initialize dataset and index into ChromaDB on startup
df_med = load_or_create_medical_df()
print(index_medical_data())

specialties_list = ["All Specialties"] + sorted(df_med["category"].unique().tolist()) + ["Uploaded Medical PDF"]

def handle_med_chat(user_message, specialty, chat_history):
    if chat_history is None:
        chat_history = []
    if not user_message or not user_message.strip():
        return "", chat_history, "⚠️ Please enter a medical question."

    try:
        reply, sources = query_medical_rag(user_message, category_filter=specialty)
    except Exception as e:
        reply = f"❌ **Backend Error:** `{str(e)}`"
        sources = traceback.format_exc()

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})
    return "", chat_history, sources

def handle_pdf_upload(files):
    if not files:
        return "⚠️ Please select a PDF file first."
    paths = [f.name if hasattr(f, "name") else f for f in files]
    return index_medical_data(extra_files=paths)

with gr.Blocks() as med_demo:
    gr.Markdown("# 🏥 MedAssist AI — Clinical FAQ & Guideline RAG Assistant")
    gr.Markdown(
        "Powered by **Core NIH / WHO Clinical FAQs**, **ChromaDB Metadata Filtering**, and **Gemini 2.5 Flash** with strict non-hallucination guardrails."
    )

    with gr.Row():
        with gr.Column(scale=3):
            specialty_dropdown = gr.Dropdown(
                choices=specialties_list,
                value="All Specialties",
                label="🩺 Filter by Medical Specialty (ChromaDB Metadata Filter)"
            )
            chatbot = gr.Chatbot(height=430, label="MedAssist Clinical Bot")
            msg_box = gr.Textbox(
                placeholder="e.g., Can I give Ibuprofen to a patient with high fever and suspected Dengue?",
                label="Ask a Medical / Clinical Question"
            )
            with gr.Row():
                submit_btn = gr.Button("Ask MedAssist", variant="primary")
                clear_btn = gr.Button("Clear Consultation")

            gr.Examples(
                examples=[
                    ["Can I take Ibuprofen or Aspirin for Dengue fever pain, and what are the warning signs?"],
                    ["My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"],
                    ["A diabetic patient's sugar dropped to 58 mg/dL and they are dizzy. What is the 15-15 rule?"],
                    ["What is the difference between a rescue inhaler and a controller inhaler for asthma?"],
                    ["Can you prescribe an antibiotic dosage for a severe migraine headache?"]
                ],
                inputs=msg_box,
                label="🧪 Click Any Clinical Test Case Below:"
            )

        with gr.Column(scale=2):
            gr.Markdown("### 🔬 Retrieved Clinical Evidence (ChromaDB)")
            source_inspector = gr.Textbox(
                label="Matched Guidelines, Source Agency & Cosine Distance",
                lines=14,
                interactive=False
            )
            gr.Markdown("### 📄 Add Custom Medical PDFs (WHO Guides / Lab Manuals)")
            pdf_uploader = gr.File(label="Upload Medical PDF", file_count="multiple", file_types=[".pdf"])
            upload_btn = gr.Button("➕ Index Medical PDF into ChromaDB")
            upload_status = gr.Textbox(
                label="Database Status",
                value="✅ 8 Core NIH/WHO Clinical Guidelines Active",
                interactive=False
            )

    msg_box.submit(handle_med_chat, [msg_box, specialty_dropdown, chatbot], [msg_box, chatbot, source_inspector])
    submit_btn.click(handle_med_chat, [msg_box, specialty_dropdown, chatbot], [msg_box, chatbot, source_inspector])
    clear_btn.click(lambda: ("", [], ""), None, [msg_box, chatbot, source_inspector])
    upload_btn.click(handle_pdf_upload, inputs=[pdf_uploader], outputs=[upload_status])

if __name__ == "__main__":
    # In VS Code locally, share=False opens cleanly on http://127.0.0.1:7860
    med_demo.launch(theme=gr.themes.Soft())