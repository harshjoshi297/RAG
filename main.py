from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task="text-generation"
)

embedding_model= HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

llm_model = ChatHuggingFace(llm = llm)


#Indexing
#Using yt api, we wil try to get transcript of any video

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)


except TranscriptsDisabled:
    print("No captions available for this video.")


#Using text splitter to get chunks of text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#Creating embeddings sand stpring in a vector store.

vector_store = FAISS.from_documents(chunks, embedding_model)


#Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#Augmentation
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)


question          = input("What is your question?")
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})


#Generation
answer = llm_model.invoke(final_prompt)
print(answer.content)