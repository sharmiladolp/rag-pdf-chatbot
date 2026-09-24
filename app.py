import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# 1. Load Groq API Key from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")

# 2. Path Setup & PDF Load
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "Doc", "Chatbot.pdf")

print("1. Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 3. Split Text
print("2. Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 4. Embeddings & VectorDB
print("3. Setting up Vector Store...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# 5. Groq LLM Setup
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.2,
    groq_api_key=groq_api_key
)

# 6. Interactive Q&A Loop
print("\n=== RAG Chatbot Ready! (Type 'exit' to quit) ===")

while True:
    user_query = input("\nAsk a question from PDF: ")

    if user_query.lower() == "exit":
        break

    # Retrieve relevant document chunks
    relevant_docs = retriever.invoke(user_query)

    context_text = "\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )

    # Build Prompt
    full_prompt = f"""Use the following context from the document to answer the question.
If you don't know the answer, say "I don't know based on the provided PDF."

Context:
{context_text}

Question: {user_query}

Answer:"""

    try:
        # Call Groq Model
        response = llm.invoke(full_prompt)

        print("\nAnswer:", response.content)

    except Exception as e:
        print("\nError calling Groq API:", e)