import os
import json
import tempfile
from langchain.docstore import InMemoryDocstore

import numpy as np
import streamlit as st
import whisper
import google.generativeai as genai
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from audio_recorder_streamlit import audio_recorder

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(page_title="Asistente NIC con RAG", page_icon="🩺", layout="wide")

# === API Key de Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# ==============================
# CSS personalizado
# ==============================
st.markdown("""
<style>
<style>
/* Ocultar elementos del sistema */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Fondo principal: suave y profesional */
.main {
    background: linear-gradient(135deg, #f0f4ff 0%, #d9e4ff 100%);
    font-family: 'Segoe UI', sans-serif;
}

/* Contenedor del título */
.title-container {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    text-align: center;
}

.title-container h1 {
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    margin: 0;
}

.title-container p {
    color: #555;
    margin-top: 0.5rem;
    font-size: 1.1rem;
}

/* Caja de entrada del chat */
.stChatFloatingInputContainer {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}

/* Mensajes del chat */
[data-testid="stChatMessageContainer"] {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    margin-bottom: 1rem;
    min-height: 500px;
    max-height: 600px;
    overflow-y: auto;
}

/* Texto del usuario */
[data-testid="stChatMessageContainer"] p {
    color: #333;
    font-size: 1rem;
    line-height: 1.6;
}

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 10px rgba(118, 75, 162, 0.3);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 12px rgba(118, 75, 162, 0.4);
}

/* Entradas de texto */
.stTextInput > div > div > input {
    border-radius: 30px;
    border: 2px solid #667eea;
    padding: 0.6rem 1rem;
    font-size: 1rem;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #eef1fb;
    border-radius: 10px;
    font-weight: 600;
    color: #333;
}

/* Scrollbar personalizado */
[data-testid="stChatMessageContainer"]::-webkit-scrollbar {
    width: 10px;
}

[data-testid="stChatMessageContainer"]::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

[data-testid="stChatMessageContainer"]::-webkit-scrollbar-track {
    background: #f1f1f1;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CARGAR VECTORSTORE COMPLETO DESDE ARCHIVOS GUARDADOS
# ==============================
import pickle
import faiss

@st.cache_resource(show_spinner=False)
def cargar_vectorstore_desde_archivos():
    # === 1. Cargar índice FAISS
    index = faiss.read_index("indice_faiss.index")

    # === 2. Cargar embeddings y metadatos
    embeddings_array = np.load("embeddings.npy")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open("chunks_con_headers.pkl", "rb") as f:
        textos = pickle.load(f)

    # === 3. Cargar modelo de embeddings (debe ser el MISMO con el que se creó el índice)
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
     
    documentos = []
    for t in textos:
        if isinstance(t, dict):
            contenido = f"[{t.get('seccion', 'Sin sección')}] {t.get('texto', '')}"
        else:
            contenido = str(t)
        documentos.append(Document(page_content=contenido))

    # 5. Reconstruir el vectorstore
    docstore_items = {}
    index_to_docstore_id = {}
    for i, doc in enumerate(documentos):
        doc_id = f"doc_{i}"
        docstore_items[doc_id] = doc
        index_to_docstore_id[i] = doc_id
    
    docstore = InMemoryDocstore(docstore_items)
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore


# ==============================
# FUNCIONES AUXILIARES
# ==============================
@st.cache_resource
def cargar_whisper():
    return whisper.load_model("small", device="cpu")

def buscar_contexto(consulta: str, k: int = 5):
    resultados = vectorstore.similarity_search_with_score(consulta, k=k)
    contexto = "\n\n".join([doc.page_content for doc, _ in resultados])

    contexto_mostrable = ""
    for i, (doc, score) in enumerate(resultados, start=1):
        contexto_mostrable += f"🔹 **Fragmento {i} (score={score:.4f})**\n{doc.page_content}\n\n"
    
    return contexto, contexto_mostrable


def consulta_llm_rag(consulta: str, contexto: str, historial: list) -> str:
    ultimos_mensajes = historial[-5:]
    historial_texto = "\n".join([
        f"{'👤 Usuario' if m['role']=='user' else '🩺 Asistente'}: {m['content']}"
        for m in ultimos_mensajes
    ])

    prompt = f"""
Eres un asistente clínico especializado en la Clasificación NIC. 
Responde de forma amable, profesional y breve.
Usa EXCLUSIVAMENTE la información recuperada de documentos y el historial reciente.

---
🧾 Historial reciente:
{historial_texto}

📚 Contexto:
{contexto}

❓ Nueva consulta:
{consulta}

---
Si no hay suficiente información, responde: "❌ No encontrado en el documento".
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

# ==============================
# INICIALIZACIÓN DE SESIÓN
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 ¡Hola! Soy tu asistente NIC. Puedes escribir tu consulta o usar el micrófono."
    }]

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = None

if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="title-container">
    <h1>🩺 Asistente NIC con RAG</h1>
    <p>Tu asistente inteligente para consultas sobre la Clasificación de Intervenciones de Enfermería</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# CARGAR VECTORSTORE
# ==============================
vectorstore = cargar_vectorstore_desde_archivos()

# ==============================
# ÁREA DE CHAT
# ==============================
with st.container():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🩺"):
                st.write(msg["content"])
                if "context" in msg:
                    with st.expander("🔍 Ver contexto utilizado", expanded=False):
                        st.markdown(f"```\n{msg['context']}\n```")

    # Procesar audio grabado
    if st.session_state.pending_audio is not None:
        with st.spinner("🎤 Transcribiendo audio..."):
            model = cargar_whisper()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(st.session_state.pending_audio)
                tmp_path = tmp.name
            result = model.transcribe(tmp_path)
            transcribed_text = result["text"].strip()
            os.unlink(tmp_path)

        st.session_state.messages.append({"role": "user", "content": transcribed_text})
        st.session_state.pending_audio = None
        st.rerun()

    # Si el último mensaje es del usuario → buscar respuesta
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_query = st.session_state.messages[-1]["content"]

        with st.spinner("🔍 Buscando información relevante..."):
            contexto, contexto_mostrable = buscar_contexto(user_query, k=3)

        with st.spinner("💭 Generando respuesta..."):
            respuesta = consulta_llm_rag(
                consulta=user_query,
                contexto=contexto,
                historial=st.session_state.messages
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": respuesta,
            "context": contexto_mostrable
        })
        st.rerun()

# ==============================
# INPUT DE TEXTO Y AUDIO
# ==============================
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.chat_input("💬 Escribe tu consulta aquí...")

with col2:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#667eea",
        icon_name="microphone",
        icon_size="2x",
        key=f"audio_recorder_{len(st.session_state.messages)}"
    )

if audio_bytes and audio_bytes != st.session_state.audio_processed:
    st.session_state.audio_processed = audio_bytes
    st.session_state.pending_audio = audio_bytes
    st.rerun()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚕️ Este sistema es solo de apoyo y no sustituye la valoración clínica profesional.")

if len(st.session_state.messages) > 1:
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 ¡Hola! Soy tu asistente NIC. Puedes escribir tu consulta o usar el micrófono."
        }]
        st.session_state.audio_processed = None
        st.session_state.pending_audio = None
        st.rerun()
