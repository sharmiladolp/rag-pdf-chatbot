import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# Load environment variables
load_dotenv()

print("API Key loaded:", bool(os.getenv("GROQ_API_KEY")))


# -----------------------------
# 1. Load PDF Documents
# -----------------------------

documents_folder = "documents"

all_documents = []

for filename in os.listdir(documents_folder):

    if filename.endswith(".pdf"):

        file_path = os.path.join(documents_folder, filename)

        print(f"Loading: {filename}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        all_documents.extend(documents)


print("\nTotal documents/pages loaded:", len(all_documents))


# Show first page
if all_documents:

    print("\nFirst page content:")
    print(all_documents[0].page_content[:1000])


# -----------------------------
# 2. Split Documents
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(all_documents)

print("\nTotal chunks created:", len(chunks))

if chunks:

    print("\nFirst chunk:")
    print(chunks[0].page_content)


# -----------------------------
# 3. Create Embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Store in ChromaDB
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nEmbeddings created and stored in ChromaDB")


# -----------------------------
# 5. Create Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 6. Create Groq LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# -----------------------------
# 7. Ask Questions
# -----------------------------

while True:

    query = input(
        "\nAsk your question (type 'exit' to quit): "
    )

    if query.lower() == "exit":

        print("Program stopped.")
        break


    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)


    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    # Create prompt
    prompt = f"""
Answer the question using only the context below.

If the answer is not available in the context,
say "I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""


    # Ask LLM
    response = llm.invoke(prompt)


    # Display answer
    print("\nFinal Answer:")
    print(response.content)
