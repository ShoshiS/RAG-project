import streamlit as st
from rag_core import generate_answer, init_clients


@st.cache_resource(show_spinner="מתחבר ל-Pinecone ו-Gemini...")
def _cached_init_clients():
    return init_clients()


st.set_page_config(page_title="צ'אט RAG - ביטוח לאומי", page_icon="💬")

st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        direction: rtl;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1,
    .stMarkdown h2, .stMarkdown h3, .stCaption, .stChatMessage,
    [data-testid="stChatMessageContent"], [data-testid="stExpander"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    textarea, input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💬 צ'אט שאלות ותשובות")
st.caption("שאלו שאלה על המסמכים, והתשובה תיווצר על בסיס הקטעים הרלוונטיים מהאינדקס.")

try:
    client, index, embeddings_model, min_relevance_score = _cached_init_clients()
except Exception as exc:
    st.error(f"שגיאה באתחול: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("מקורות"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['source']}** (score: {src['score']:.3f})\n\n{src['text']}"
                    )

prompt = st.chat_input("כתבו את שאלתכם כאן...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("מחפש ומנסח תשובה..."):
            result = generate_answer(
                prompt,
                client,
                index,
                embeddings_model,
                min_relevance_score,
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("מקורות"):
                for src in result["sources"]:
                    st.markdown(
                        f"**{src['source']}** (score: {src['score']:.3f})\n\n{src['text']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
