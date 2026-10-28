import traceback
import gradio as gr
from rag_chain import CLINICAL_DOCS, ask_med_langchain, index_uploaded_pdfs

specialties = ["All Specialties"] + sorted(list({d.metadata["category"] for d in CLINICAL_DOCS})) + ["Uploaded Medical PDF"]

def handle_chat(user_message, specialty, chat_history):
    chat_history = chat_history or []
    if not user_message or not user_message.strip():
        return "", chat_history, "⚠️ Please enter a question."
    try:
        reply, sources = ask_med_langchain(user_message, specialty)
    except Exception as e:
        reply, sources = f"❌ **Error:** `{e}`", traceback.format_exc()

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply})
    return "", chat_history, sources

def handle_pdf(files):
    if not files:
        return "⚠️ Select a PDF first."
    return index_uploaded_pdfs([f.name if hasattr(f, "name") else f for f in files])

with gr.Blocks() as demo:
    gr.Markdown("# 🏥 MedAssist AI — LangChain Clinical RAG")
    with gr.Row():
        with gr.Column(scale=3):
            spec_drop = gr.Dropdown(specialties, value="All Specialties", label="🩺 Filter by Specialty")
            chatbot = gr.Chatbot(height=420, label="MedAssist (LangChain LCEL)")
            msg = gr.Textbox(placeholder="Ask a clinical question...", label="Your Question")
            with gr.Row():
                btn = gr.Button("Ask MedAssist", variant="primary")
                clr = gr.Button("Clear")
            gr.Examples(
                examples=[
                    ["Can I take Ibuprofen for Dengue fever pain, and what are the warning signs?"],
                    ["My fasting blood sugar is 118 mg/dL and HbA1c is 6.1%. Do I have diabetes?"],
                    ["Can you prescribe an antibiotic dosage for a severe migraine?"]
                ],
                inputs=msg
            )
        with gr.Column(scale=2):
            inspector = gr.Textbox(label="🔬 Retrieved Chunks (LangChain Chroma)", lines=13, interactive=False)
            pdfs = gr.File(label="📄 Upload Medical PDF", file_count="multiple", file_types=[".pdf"])
            up_btn = gr.Button("➕ Index PDF via PyPDFLoader")
            status = gr.Textbox(label="Status", value="✅ 8 Clinical Docs Loaded", interactive=False)

    msg.submit(handle_chat, [msg, spec_drop, chatbot], [msg, chatbot, inspector])
    btn.click(handle_chat, [msg, spec_drop, chatbot], [msg, chatbot, inspector])
    clr.click(lambda: ("", [], ""), None, [msg, chatbot, inspector])
    up_btn.click(handle_pdf, [pdfs], [status])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())