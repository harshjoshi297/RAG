import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse, parse_qs

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
try:
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    if "SUPADATA_API_KEY" in st.secrets:
        os.environ["SUPADATA_API_KEY"] = st.secrets["SUPADATA_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    st.error("HUGGINGFACEHUB_API_TOKEN not found. Check your .env file or Streamlit secrets.")
    st.stop()

if not os.environ.get("SUPADATA_API_KEY"):
    st.error("SUPADATA_API_KEY not found. Check your .env file or Streamlit secrets.")
    st.stop()

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Check your .env file or Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube Video Chatbot")
st.caption("Ask questions about any YouTube video using AI!")

# --- HELPER FUNCTIONS ---
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[1].split("/")[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/").split("?")[0]
    raise ValueError(f"Could not extract video ID from URL: {url}")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    llm_model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=512
    )
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    return llm_model, embedding_model


@st.cache_resource(show_spinner="Fetching and indexing transcript...", hash_funcs={str: lambda x: x})
def build_rag_chain(video_id: str, _llm_model, _embedding_model):

    # Fetch transcript using Supadata API
    response = requests.get(
        f"https://api.supadata.ai/v1/youtube/transcript?videoId={video_id}",
        headers={"x-api-key": os.environ["SUPADATA_API_KEY"]}
    )
    data = response.json()

    if "content" not in data:
        raise ValueError(f"Could not fetch transcript. API response: {data}")

    transcript = " ".join([item["text"] for item in data["content"]])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, _embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm_model
        | StrOutputParser()
    )
    return rag_chain


# --- UI ---
youtube_url = st.text_input("Enter a YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if youtube_url:
    try:
        video_id = extract_video_id(youtube_url)

        # Clear chat history if video changes
        if "current_video_id" not in st.session_state or st.session_state.current_video_id != video_id:
            st.session_state.messages = []
            st.session_state.current_video_id = video_id

        st.success(f"Video ID extracted: `{video_id}`")
        st.video(youtube_url)

        llm_model, embedding_model = load_models()
        rag_chain = build_rag_chain(video_id, llm_model, embedding_model)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_query := st.chat_input("Ask a question about the video..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = rag_chain.invoke(user_query)
                        if not response or response.strip() == "":
                            st.warning("Model returned an empty response. Try rephrasing your question.")
                        else:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

    except ValueError as e:
        st.error(f"Invalid URL: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
