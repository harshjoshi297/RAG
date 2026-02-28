import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse, parse_qs

load_dotenv()

try:
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    pass

if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    st.error("HUGGINGFACEHUB_API_TOKEN not found.")
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
    llm = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-7B-Instruct-v0.3',
        task="text-generation",
        max_new_tokens=512
    )
    llm_model = ChatHuggingFace(llm=llm)
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    return llm_model, embedding_model


@st.cache_resource(show_spinner="Fetching and indexing transcript...", hash_funcs={str: lambda x: x})
def build_rag_chain(video_id: str, _llm_model, _embedding_model):
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB", "en-CA", "en-AU"])
    except Exception:
        transcript_list = ytt_api.fetch(video_id)

    transcript = " ".join(chunk.text for chunk in transcript_list)

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
        st.success(f"Video ID extracted: `{video_id}`")
        st.video(youtube_url)

        llm_model, embedding_model = load_models()
        rag_chain = build_rag_chain(video_id, llm_model, embedding_model)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

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
                    response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    except ValueError as e:
        st.error(f"Invalid URL: {e}")
    except Exception as e:
        st.error(f"Error: {e}")